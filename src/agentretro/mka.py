from __future__ import annotations

import gzip
import heapq
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from abc import ABC, abstractmethod
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

import numpy as np
import pandas as pd
from rdkit import Chem, RDConfig
from rdkit import DataStructs

from .chemistry import (
    canonical_components,
    canonical_smiles,
    druglike_properties,
    morgan_fingerprint,
    murcko_scaffold,
)
from .io import atomic_json, require_free_space, sha256_file, stable_hash, write_table

try:
    import sys

    sys.path.append(str(Path(RDConfig.RDContribDir) / "SA_Score"))
    import sascorer
except Exception:
    sascorer = None

MODEL_ID = "Qwen/Qwen3-8B-AWQ"
MODEL_REVISION = "4da05a8edb55c6046cce958586c33b61da07bb79"
SCIBERT_ID = "allenai/scibert_scivocab_uncased"
SCIBERT_REVISION = "24f92d32b1bfb0bcaf9ab193ff3ad01e87732fc1"
DESIGN_STRATEGIES = {
    "scaffold transition",
    "functional-group interconversion",
    "side-chain modification",
}


def interleave_template_frequencies(counts: pd.Series) -> list[str]:
    rare = counts.sort_values(ascending=True, kind="stable").index.astype(str).tolist()
    common = counts.sort_values(ascending=False, kind="stable").index.astype(str).tolist()
    result: list[str] = []
    seen: set[str] = set()
    for low, high in zip(rare, common):
        for template in (low, high):
            if template not in seen:
                seen.add(template)
                result.append(template)
    result.extend(template for template in rare if template not in seen)
    return result


def select_frequency_matched_template(
    candidates: Sequence[tuple[str, str]],
    assigned: Counter[str],
    target_probabilities: Mapping[str, float],
    next_total: int,
) -> tuple[str, str]:
    best_by_template: dict[str, tuple[float, str]] = {}
    for template, reactants in candidates:
        try:
            accessibility = synthetic_accessibility(reactants.split(".")[0])
        except (ValueError, RuntimeError):
            accessibility = float("inf")
        value = (accessibility, reactants)
        if template not in best_by_template or value < best_by_template[template]:
            best_by_template[template] = value
    if not best_by_template:
        raise ValueError("No applicable template candidates")
    template = min(
        best_by_template,
        key=lambda value: (
            assigned[value] - target_probabilities.get(value, 0.0) * next_total,
            best_by_template[value][0],
            value,
            best_by_template[value][1],
        ),
    )
    return template, best_by_template[template][1]


def normalize_design_strategy(value: str) -> str:
    normalized = " ".join(str(value).strip().lower().replace("_", " ").split())
    aliases = {
        "functional group interconversion": "functional-group interconversion",
        "side chain modification": "side-chain modification",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in DESIGN_STRATEGIES:
        raise ValueError(f"Unsupported MKA design strategy: {value!r}")
    return normalized


def synthetic_accessibility(smiles: str) -> float:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(smiles)
    if sascorer is None:
        rings = mol.GetRingInfo().NumRings()
        stereo = len(Chem.FindMolChiralCenters(mol, includeUnassigned=True))
        return float(min(10.0, 1.0 + 0.04 * mol.GetNumHeavyAtoms() + 0.3 * rings + 0.2 * stereo))
    return float(sascorer.calculateScore(mol))


def _iter_chembl(path: str | Path) -> Iterator[dict[str, str]]:
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8", errors="replace") as handle:
        header = handle.readline().rstrip("\n").split("\t")
        lower = [x.lower() for x in header]
        id_idx = lower.index("chembl_id") if "chembl_id" in lower else 0
        smiles_candidates = ["canonical_smiles", "canonical_smiles_standardized"]
        smiles_idx = next((lower.index(x) for x in smiles_candidates if x in lower), 1)
        for line in handle:
            fields = line.rstrip("\n").split("\t")
            if len(fields) > max(id_idx, smiles_idx):
                yield {"chembl_id": fields[id_idx], "smiles": fields[smiles_idx]}


def deterministic_chembl_sample(path: str | Path, sample_size: int = 100_000) -> list[dict[str, str]]:
    heap: list[tuple[int, str, str]] = []
    for row in _iter_chembl(path):
        key = int(stable_hash(row["chembl_id"])[:16], 16)
        item = (-key, row["chembl_id"], row["smiles"])
        if len(heap) < sample_size:
            heapq.heappush(heap, item)
        elif item > heap[0]:
            heapq.heapreplace(heap, item)
    return [
        {"chembl_id": chembl_id, "smiles": smiles}
        for _, chembl_id, smiles in sorted(heap, key=lambda x: -x[0])
    ]


def screen_candidates(
    candidates: Iterable[dict[str, str]],
    train_products: Sequence[str],
    keep: int = 12_000,
    similarity_cutoff: float = 0.7,
) -> pd.DataFrame:
    train_fps = [morgan_fingerprint(x) for x in train_products]
    accepted: list[dict[str, Any]] = []
    seen: set[str] = set()
    audit = {
        "attempted": 0,
        "invalid_or_nonorganic": 0,
        "duplicate_structure": 0,
        "property_filter": 0,
        "similarity_filter": 0,
        "eligible": 0,
    }
    for row in candidates:
        audit["attempted"] += 1
        try:
            smiles = canonical_smiles(row["smiles"])
            if "." in smiles:
                audit["invalid_or_nonorganic"] += 1
                continue
            mol = Chem.MolFromSmiles(smiles)
            if (
                mol is None
                or not any(atom.GetAtomicNum() == 6 for atom in mol.GetAtoms())
                or any(atom.GetAtomicNum() == 0 for atom in mol.GetAtoms())
            ):
                audit["invalid_or_nonorganic"] += 1
                continue
            if smiles in seen:
                audit["duplicate_structure"] += 1
                continue
            seen.add(smiles)
            props = druglike_properties(smiles)
            sa = synthetic_accessibility(smiles)
            if not (150 <= props["mw"] <= 700 and props["qed"] >= 0.35 and sa <= 6):
                audit["property_filter"] += 1
                continue
            fp = morgan_fingerprint(smiles)
            max_similarity = max(DataStructs.BulkTanimotoSimilarity(fp, train_fps), default=0.0)
            if max_similarity >= similarity_cutoff:
                audit["similarity_filter"] += 1
                continue
            accepted.append(
                {
                    **row,
                    "smiles": smiles,
                    **props,
                    "sa_score": sa,
                    "max_train_tanimoto": max_similarity,
                }
            )
            audit["eligible"] += 1
        except (ValueError, RuntimeError):
            audit["invalid_or_nonorganic"] += 1
            continue
    accepted.sort(key=lambda x: (-x["qed"], x["sa_score"], x["max_train_tanimoto"], x["chembl_id"]))
    result = pd.DataFrame(accepted[:keep])
    audit["retained"] = len(result)
    audit["requested_keep"] = int(keep)
    audit["similarity_cutoff"] = float(similarity_cutoff)
    result.attrs["screening_audit"] = audit
    return result


class GenerationEngine(ABC):
    @abstractmethod
    def generate(self, prompts: Sequence[str]) -> list[str]:
        raise NotImplementedError


class VLLMEngine(GenerationEngine):
    def __init__(self, model_path: str = MODEL_ID, seed: int = 2026):
        project = Path(__file__).resolve().parents[2]
        os.environ.setdefault("XDG_CACHE_HOME", str(project / "artifacts/cache"))
        os.environ.setdefault("HF_HOME", str(project / "artifacts/cache/huggingface"))
        os.environ.setdefault("VLLM_CACHE_ROOT", str(project / "artifacts/cache/vllm"))
        from transformers import AutoTokenizer

        local = project / "artifacts/weights/qwen3-8b-awq"
        source = str(local) if model_path == MODEL_ID and (local / "config.json").exists() else model_path
        revision = None if source != model_path else MODEL_REVISION
        self.tokenizer = AutoTokenizer.from_pretrained(source, revision=revision)
        self.seed = seed
        self.generation_calls = 0
        self.source = source
        try:
            from vllm import LLM, SamplingParams

            self.backend = "vllm"

            self.model = LLM(
                model=source,
                quantization="awq",
                seed=seed,
                trust_remote_code=False,
                max_model_len=2048,
                max_num_seqs=64,
                gpu_memory_utilization=0.60,
            )
            self.sampling_params_class = SamplingParams
        except ImportError:
            import torch
            from transformers import AutoModelForCausalLM

            self.backend = "transformers"
            self.model = AutoModelForCausalLM.from_pretrained(
                source,
                revision=revision,
                device_map="auto",
                torch_dtype=torch.float16,
                trust_remote_code=False,
            ).eval()

    def generate(self, prompts: Sequence[str]) -> list[str]:
        call_seed = self.seed + self.generation_calls
        self.generation_calls += 1
        rendered = [
            self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            for prompt in prompts
        ]
        if self.backend == "vllm":
            from vllm.sampling_params import GuidedDecodingParams

            guided = None
            if prompts and all("product_smiles" in prompt for prompt in prompts):
                guided = GuidedDecodingParams(
                    json={
                        "type": "object",
                        "properties": {
                            "product_smiles": {"type": "string", "maxLength": 256},
                            "strategy": {
                                "type": "string",
                                "enum": sorted(DESIGN_STRATEGIES),
                            },
                            "design_principle": {"type": "string", "maxLength": 256},
                        },
                        "required": ["product_smiles", "strategy", "design_principle"],
                        "additionalProperties": False,
                    }
                )
            elif prompts and all("reaction_principle" in prompt for prompt in prompts):
                guided = GuidedDecodingParams(
                    json={
                        "type": "object",
                        "properties": {
                            "reaction_principle": {"type": "string", "maxLength": 256},
                            "product_structure": {"type": "string", "maxLength": 256},
                            "reactant_properties": {"type": "string", "maxLength": 256},
                            "likely_reaction_family": {"type": "string", "maxLength": 256},
                        },
                        "required": [
                            "reaction_principle",
                            "product_structure",
                            "reactant_properties",
                            "likely_reaction_family",
                        ],
                        "additionalProperties": False,
                    }
                )
            params = [
                self.sampling_params_class(
                    temperature=0.2,
                    top_p=0.8,
                    max_tokens=1024,
                    seed=(self.seed + int(stable_hash(prompt)[:8], 16)) % (2**31 - 1),
                    stop=["}"],
                    include_stop_str_in_output=True,
                    guided_decoding=guided,
                )
                for prompt in prompts
            ]
            outputs = self.model.generate(rendered, params, use_tqdm=False)
            return [item.outputs[0].text for item in outputs]
        import torch

        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        tokens = self.tokenizer(rendered, return_tensors="pt", padding=True)
        device = next(self.model.parameters()).device
        tokens = {key: value.to(device) for key, value in tokens.items()}
        torch.manual_seed(call_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(call_seed)
        with torch.no_grad():
            generated = self.model.generate(
                **tokens, do_sample=True, temperature=0.2, top_p=0.8,
                max_new_tokens=1024, pad_token_id=self.tokenizer.pad_token_id,
                stop_strings=["}"], tokenizer=self.tokenizer,
            )
        prompt_width = tokens["input_ids"].shape[1]
        return [
            self.tokenizer.decode(row[prompt_width:], skip_special_tokens=True)
            for row in generated
        ]


def parse_json_response(value: str) -> dict[str, Any]:
    value = value.strip()
    if value.startswith("```"):
        value = value.split("\n", 1)[-1].rsplit("```", 1)[0]
    start, end = value.find("{"), value.rfind("}")
    if start < 0 or end < start:
        raise ValueError("No JSON object in model response")

    candidate = value[start : end + 1].replace("\x08", "\\\\")
    parsed = json.loads(candidate, strict=False)
    if not isinstance(parsed, dict):
        raise ValueError("Expected a JSON object")
    return parsed


def validate_json_payload(prompt: str, payload: dict[str, Any]) -> dict[str, Any]:
    required: tuple[str, ...] = ()
    if "product_smiles" in prompt and "design_principle" in prompt:
        required = ("product_smiles", "strategy", "design_principle")
    elif "reaction_principle" in prompt and "likely_reaction_family" in prompt:
        required = (
            "reaction_principle",
            "product_structure",
            "reactant_properties",
            "likely_reaction_family",
        )
    missing = [key for key in required if not isinstance(payload.get(key), str) or not payload[key].strip()]
    if missing:
        raise ValueError(f"JSON response has missing or empty required fields: {missing}")
    return payload


def generate_json_with_retry(
    engine: GenerationEngine,
    prompt: str,
    cache_dir: str | Path,
    cache_key: str,
    retries: int = 3,
) -> tuple[dict[str, Any] | None, str | None]:
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    last_error: str | None = None
    for attempt in range(1, retries + 1):
        cached = cache_dir / f"{cache_key}.attempt-{attempt}.json"
        if cached.exists():
            with cached.open("r", encoding="utf-8") as handle:
                saved = json.load(handle)
            expected_prompt = (
                prompt
                if attempt == 1
                else prompt + f"\nJSON retry {attempt}: return one complete valid JSON object only."
            )
            if saved.get("prompt") != expected_prompt:
                raise RuntimeError(
                    f"Generation cache-key collision for {cached.name}: cached prompt differs"
                )
            if saved.get("error") is None:
                try:
                    payload = parse_json_response(saved["response"])
                    return validate_json_payload(prompt, payload), saved["response"]
                except (ValueError, json.JSONDecodeError, TypeError):
                    pass
            last_error = saved.get("error")
            continue
        actual_prompt = (
            prompt
            if attempt == 1
            else prompt + f"\nJSON retry {attempt}: return one complete valid JSON object only."
        )
        response = engine.generate([actual_prompt])[0]
        payload: dict[str, Any] | None = None
        last_error = None
        try:
            payload = validate_json_payload(prompt, parse_json_response(response))
        except (ValueError, json.JSONDecodeError, TypeError) as exc:
            last_error = f"{type(exc).__name__}: {exc}"
        atomic_json(
            cache_dir / f"{cache_key}.attempt-{attempt}.json",
            {"prompt": actual_prompt, "response": response, "error": last_error},
        )
        if payload is not None:
            return payload, response
    return None, None


def generate_json_batch_with_retry(
    engine: GenerationEngine,
    prompts: Sequence[str],
    cache_dir: str | Path,
    cache_keys: Sequence[str],
    retries: int = 3,
) -> list[dict[str, Any] | None]:
    if len(prompts) != len(cache_keys):
        raise ValueError("prompts and cache_keys differ in length")
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any] | None] = [None] * len(prompts)
    pending = list(range(len(prompts)))
    for attempt in range(1, retries + 1):
        uncached = []
        cached_responses: dict[int, str] = {}
        for index in pending:
            path = cache_dir / f"{cache_keys[index]}.attempt-{attempt}.json"
            if path.exists():
                with path.open("r", encoding="utf-8") as handle:
                    saved = json.load(handle)
                expected_prompt = (
                    prompts[index]
                    if attempt == 1
                    else prompts[index]
                    + f"\nJSON retry {attempt}: return one complete valid JSON object only."
                )
                if saved.get("prompt") != expected_prompt:
                    raise RuntimeError(
                        f"Generation cache-key collision for {path.name}: cached prompt differs"
                    )
                cached_responses[index] = saved.get("response", "")
            else:
                uncached.append(index)
        actual_prompts = {
            index: (
                prompts[index]
                if attempt == 1
                else prompts[index]
                + f"\nJSON retry {attempt}: return one complete valid JSON object only."
            )
            for index in uncached
        }
        generated = (
            engine.generate([actual_prompts[index] for index in uncached]) if uncached else []
        )
        response_by_index = {index: response for index, response in zip(uncached, generated)}
        response_by_index.update(cached_responses)
        next_pending = []
        for index in pending:
            response = response_by_index[index]
            error = None
            try:
                results[index] = validate_json_payload(prompts[index], parse_json_response(response))
            except (ValueError, json.JSONDecodeError, TypeError) as exc:
                error = f"{type(exc).__name__}: {exc}"
                next_pending.append(index)
            cache_path = cache_dir / f"{cache_keys[index]}.attempt-{attempt}.json"
            if not cache_path.exists():
                atomic_json(
                    cache_path,
                    {
                        "prompt": actual_prompts[index],
                        "response": response,
                        "error": error,
                    },
                )
        pending = next_pending
        if not pending:
            break
    return results


def sketch_prompt(seed_smiles: str) -> str:
    return f"""You are a medicinal chemistry molecule designer. Analyze the seed molecule
{seed_smiles}
and create one chemically valid, synthetically accessible product that is structurally different.
Use exactly one strategy: scaffold transition, functional-group interconversion, or side-chain modification.
Do not include salts, mixtures, atom maps, prose outside JSON, or a molecule identical to the seed.
Return exactly: {{"product_smiles":"...","strategy":"...","design_principle":"..."}}"""


def rationale_prompt(product: str, reactants: str) -> str:
    return f"""Explain this retrosynthetic disconnection for model training.
Product: {product}
Reactants: {reactants}
Return JSON only with keys reaction_principle, product_structure, reactant_properties, likely_reaction_family.
Be concise, factual, and do not invent reaction conditions."""


def pubchem_identity_exists(smiles: str, retries: int = 3) -> bool | None:
    encoded = urllib.parse.quote(smiles, safe="")
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/fastidentity/smiles/{encoded}/cids/JSON"
    for attempt in range(retries):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "AgentRetro-Reproduction/0.1"})
            with urllib.request.urlopen(request, timeout=45) as response:
                payload = json.load(response)
            return bool(payload.get("IdentifierList", {}).get("CID", []))
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                return False
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
            pass
        time.sleep(2**attempt)
    return None


def cached_pubchem_identity(smiles: str, cache_dir: str | Path) -> bool | None:
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{stable_hash(smiles)[:24]}.json"
    if path.exists():
        with path.open("r", encoding="utf-8") as handle:
            cached = json.load(handle)
        if cached.get("exists") is not None:
            return bool(cached["exists"])
    exists = pubchem_identity_exists(smiles)
    atomic_json(path, {"smiles": smiles, "exists": exists})
    return exists


class RetroTemplateLibrary:
    def __init__(self, templates: Sequence[str]):
        self.templates = list(templates)
        self._queries = []
        self._reactions: dict[int, Any] = {}
        for template in self.templates:
            product_smarts = template.split(">>", 1)[0]
            self._queries.append(Chem.MolFromSmarts(product_smarts))

    def apply(self, product: str, limit: int = 64) -> list[tuple[str, str]]:
        from .chemistry import _template_module

        _template_module()
        from rdchiral.initialization import rdchiralReaction, rdchiralReactants
        from rdchiral.main import rdchiralRun

        molecule = Chem.MolFromSmiles(product)
        if molecule is None:
            return []
        reactant = rdchiralReactants(product)
        outputs: list[tuple[str, str]] = []
        for index, (template, query) in enumerate(zip(self.templates, self._queries)):
            if query is None or not molecule.HasSubstructMatch(query):
                continue
            try:
                reaction = self._reactions.get(index)
                if reaction is None:
                    reaction = rdchiralReaction(template)
                    self._reactions[index] = reaction
                for result in rdchiralRun(reaction, reactant):
                    canonical = canonical_components(result)
                    outputs.append((template, canonical))
                    if len(outputs) >= limit:
                        return outputs
            except Exception:
                continue
        return outputs


def apply_retro_templates(
    product: str,
    templates: Sequence[str] | RetroTemplateLibrary,
    limit: int = 64,
) -> list[tuple[str, str]]:
    if isinstance(templates, RetroTemplateLibrary):
        return templates.apply(product, limit=limit)
    from .chemistry import _template_module

    _template_module()
    from rdchiral.initialization import rdchiralReaction, rdchiralReactants
    from rdchiral.main import rdchiralRun

    reactant = rdchiralReactants(product)
    outputs: list[tuple[str, str]] = []
    for template in templates:
        try:
            reaction = rdchiralReaction(template)
            for result in rdchiralRun(reaction, reactant):
                canonical = canonical_components(result)
                outputs.append((template, canonical))
                if len(outputs) >= limit:
                    return outputs
        except Exception:
            continue
    return outputs


@dataclass
class MKAStats:
    attempted: int = 0
    invalid_generation: int = 0
    property_filter: int = 0
    duplicate_generation: int = 0
    no_reactants: int = 0
    test_leakage: int = 0
    pubchem_known: int = 0
    pubchem_unknown: int = 0
    accepted: int = 0


def generate_pseudo_ood(
    seeds: pd.DataFrame,
    train: pd.DataFrame,
    ood_test: pd.DataFrame,
    engine: GenerationEngine,
    output_dir: str | Path,
    target_count: int = 4_000,
    batch_size: int = 64,
    strict: bool = True,
    design_attempt_limit: int = 3,
) -> pd.DataFrame:
    output_dir = Path(output_dir)
    raw_dir = output_dir / "raw_responses"
    pubchem_dir = output_dir / "pubchem_identity"
    raw_dir.mkdir(parents=True, exist_ok=True)
    template_counts = train["retro_template"].astype(str).value_counts()
    templates = interleave_template_frequencies(template_counts)
    template_probabilities = (template_counts / template_counts.sum()).to_dict()
    template_library = RetroTemplateLibrary(templates)
    template_to_minimal = (
        train.groupby("retro_template")["minimal_template"]
        .agg(lambda values: values.value_counts().index[0])
        .to_dict()
    )
    test_templates = set(ood_test["retro_template"])
    test_scaffolds = set(ood_test["scaffold"])
    known_products = set(train["product"]) | set(ood_test["product"])
    partial_path = output_dir / "pseudo_ood.partial.parquet"
    progress_path = output_dir / "generation_progress.json"
    final_path = output_dir / "pseudo_ood.parquet"
    audit_path = output_dir / "generation_audit.json"
    if final_path.exists() and audit_path.exists():
        existing_audit = json.loads(audit_path.read_text())
        if existing_audit.get("accepted") == target_count:
            return pd.read_parquet(final_path)

    accepted: list[dict[str, Any]] = []
    stats = MKAStats()
    start_index = 0
    start_attempt = 0
    if partial_path.exists() and progress_path.exists():
        progress = json.loads(progress_path.read_text())
        if int(progress.get("target_count", -1)) != target_count:
            raise RuntimeError("Existing MKA progress uses a different target_count")
        partial = pd.read_parquet(partial_path)
        accepted = partial.to_dict(orient="records")
        stats = MKAStats(**progress["stats"])
        start_index = int(progress["next_seed_index"])
        start_attempt = int(progress.get("design_attempt", 0))
    seen: set[str] = {row["product"] for row in accepted}
    accepted_seed_ids = {
        json.loads(row["provenance"])["seed_chembl_id"] for row in accepted
    }
    assigned_templates: Counter[str] = Counter(
        str(row["retro_template"]) for row in accepted
    )
    seed_rows = seeds.to_dict(orient="records")
    for design_attempt in range(start_attempt, design_attempt_limit):
        attempt_start = start_index if design_attempt == start_attempt else 0
        for start in range(attempt_start, len(seed_rows), batch_size):
            require_free_space(output_dir, 8.0)
            if len(accepted) >= target_count:
                break
            raw_batch = seed_rows[start : start + batch_size]
            batch = [row for row in raw_batch if row["chembl_id"] not in accepted_seed_ids]
            retry_instruction = (
                ""
                if design_attempt == 0
                else f"\nRetry {design_attempt + 1}: the prior proposal failed automated validation. "
                "Return a different product structure."
            )
            prompts = [sketch_prompt(row["smiles"]) + retry_instruction for row in batch]
            cache_keys = [
                stable_hash(row["chembl_id"] + prompt)[:24]
                for row, prompt in zip(batch, prompts)
            ]
            payloads = generate_json_batch_with_retry(engine, prompts, raw_dir, cache_keys)
            prepared: list[dict[str, Any]] = []
            pending_products: set[str] = set()
            for row, prompt, cache_key, payload in zip(batch, prompts, cache_keys, payloads):
                stats.attempted += 1
                try:
                    if payload is None:
                        raise ValueError("generation exhausted JSON retries")
                    product = canonical_smiles(str(payload["product_smiles"]))
                    if product == row["smiles"] or "." in product:
                        raise ValueError("unchanged or multicomponent product")
                except Exception:
                    stats.invalid_generation += 1
                    continue
                try:
                    mol = Chem.MolFromSmiles(product)
                    props = druglike_properties(product)
                    if (
                        mol is None
                        or not any(atom.GetAtomicNum() == 6 for atom in mol.GetAtoms())
                        or not (150 <= props["mw"] <= 700)
                        or props["qed"] < 0.35
                        or synthetic_accessibility(product) > 6
                    ):
                        stats.property_filter += 1
                        continue
                except (ValueError, RuntimeError):
                    stats.property_filter += 1
                    continue
                if product in seen or product in pending_products or product in known_products:
                    stats.duplicate_generation += 1
                    continue
                scaffold = murcko_scaffold(product)
                candidates = apply_retro_templates(product, template_library, limit=128)
                candidates = [
                    (template, reactants)
                    for template, reactants in candidates
                    if template not in test_templates
                ]
                if not candidates:
                    stats.no_reactants += 1
                    continue
                if scaffold in test_scaffolds:
                    stats.test_leakage += 1
                    continue
                pending_products.add(product)
                prepared.append(
                    {
                        "seed": row,
                        "cache_key": cache_key,
                        "product": product,
                        "scaffold": scaffold,
                        "candidates": candidates,
                        "design_strategy": normalize_design_strategy(payload["strategy"]),
                        "design_strategy_raw": str(payload["strategy"]),
                        "design_principle": str(payload["design_principle"]),
                        "design_attempt": design_attempt + 1,
                    }
                )

            with ThreadPoolExecutor(max_workers=min(8, max(1, len(prepared)))) as pool:
                identities = list(
                    pool.map(
                        lambda item: cached_pubchem_identity(item["product"], pubchem_dir),
                        prepared,
                    )
                )
            novel: list[dict[str, Any]] = []
            for item, exists in zip(prepared, identities):
                if exists is None:
                    stats.pubchem_unknown += 1
                    continue
                if exists:
                    stats.pubchem_known += 1
                    continue
                novel.append(item)

            for item in novel:
                template, reactants = select_frequency_matched_template(
                    item.pop("candidates"),
                    assigned_templates,
                    template_probabilities,
                    sum(assigned_templates.values()) + 1,
                )
                item["template"] = template
                item["reactants"] = reactants
                assigned_templates[template] += 1

            rationale_prompts = [
                rationale_prompt(item["product"], item["reactants"]) for item in novel
            ]
            rationale_keys = [
                item["cache_key"] + ".rationale." + stable_hash(prompt)[:12]
                for item, prompt in zip(novel, rationale_prompts)
            ]
            rationale_payloads = (
                generate_json_batch_with_retry(engine, rationale_prompts, raw_dir, rationale_keys)
                if novel
                else []
            )
            for item, rationale, rationale_key in zip(novel, rationale_payloads, rationale_keys):
                if rationale is None:
                    assigned_templates[item["template"]] -= 1
                    stats.invalid_generation += 1
                    continue
                product = item["product"]
                template = item["template"]
                reactants = item["reactants"]
                seen.add(product)
                accepted_seed_ids.add(item["seed"]["chembl_id"])
                accepted.append(
                    {
                        "reaction_id": f"mka-{len(accepted):06d}",
                        "product": product,
                        "reactants": reactants,
                        "minimal_template": template_to_minimal[template],
                        "retro_template": template,
                        "scaffold": item["scaffold"],
                        "minimal_template_id": stable_hash(template_to_minimal[template])[:16],
                        "retro_template_id": stable_hash(template)[:16],
                        "scaffold_id": stable_hash(item["scaffold"])[:16],
                        "design_strategy": item["design_strategy"],
                        "design_strategy_raw": item["design_strategy_raw"],
                        "design_principle": item["design_principle"],
                        "split": "train",
                        "domain_id": 0,
                        "rationale": json.dumps(rationale, sort_keys=True),
                        "provenance": json.dumps(
                            {
                                "source": "MKA",
                                "seed_chembl_id": item["seed"]["chembl_id"],
                                "design_attempt": item["design_attempt"],
                                "model": MODEL_ID,
                                "revision": MODEL_REVISION,
                                "rationale_cache_key": rationale_key,
                            },
                            sort_keys=True,
                        ),
                    }
                )
                stats.accepted += 1
                if len(accepted) >= target_count:
                    break
            write_table(pd.DataFrame(accepted), partial_path)
            atomic_json(
                progress_path,
                {
                    "status": "complete" if len(accepted) >= target_count else "running",
                    "target_count": int(target_count),
                    "design_attempt": int(design_attempt),
                    "next_seed_index": int(start + len(raw_batch)),
                    "stats": stats.__dict__,
                },
            )
        if len(accepted) >= target_count:
            break
        start_index = 0
        atomic_json(
            progress_path,
            {
                "status": "running",
                "target_count": int(target_count),
                "design_attempt": int(design_attempt + 1),
                "next_seed_index": 0,
                "stats": stats.__dict__,
            },
        )
    result = pd.DataFrame(accepted)
    write_table(result, final_path)
    result_templates = set(result["retro_template"]) if not result.empty else set()
    result_scaffolds = set(result["scaffold"]) if not result.empty else set()
    generation_audit = {
        **stats.__dict__,
        "target_count": int(target_count),
        "design_attempt_limit": int(design_attempt_limit),
        "accepted_unique_seeds": len(accepted_seed_ids),
        "strict": bool(strict),
        "model": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "generation_backend": getattr(engine, "backend", type(engine).__name__),
        "decoding": {
            "thinking": False,
            "temperature": 0.2,
            "top_p": 0.8,
            "seed": 2026,
            "seed_derivation": "2026 + first-32-bit SHA256(prompt), modulo 2^31-1",
            "max_new_tokens": 1024,
            "json_constrained": getattr(engine, "backend", None) == "vllm",
        },
        "unique_products": int(result["product"].nunique()) if not result.empty else 0,
        "design_metadata": {
            "source": "accepted Qwen response",
            "rows": len(result),
            "strategy_counts": {
                str(key): int(value)
                for key, value in (
                    result["design_strategy"].value_counts().sort_index().items()
                    if "design_strategy" in result else []
                )
            },
        },
        "radius1_template_overlap_with_ood": len(result_templates & test_templates),
        "murcko_scaffold_overlap_with_ood": len(result_scaffolds & test_scaffolds),
        "pubchem_network_failures_accepted": 0,
        "template_assignment": {
            "enumeration_order": "interleaved_rare_common_training_frequency",
            "candidate_output_limit": 128,
            "selection": "minimum assigned-minus-real-train-expected count, then SA",
            "unique_templates": int(result["retro_template"].nunique()) if not result.empty else 0,
            "top1_fraction": float(result["retro_template"].value_counts(normalize=True).iloc[0]) if not result.empty else 0.0,
            "top10_fraction": float(result["retro_template"].value_counts(normalize=True).iloc[:10].sum()) if not result.empty else 0.0,
        },
    }
    atomic_json(audit_path, generation_audit)
    if generation_audit["radius1_template_overlap_with_ood"]:
        raise AssertionError("pseudo-OOD radius-1 template leakage")
    if generation_audit["murcko_scaffold_overlap_with_ood"]:
        raise AssertionError("pseudo-OOD scaffold leakage")
    if strict and len(result) != target_count:
        raise RuntimeError(f"MKA accepted {len(result)} reactions; formal target is {target_count}")
    if len(result) == target_count:
        partial_path.unlink(missing_ok=True)
    return result


def generate_training_rationales(
    frame: pd.DataFrame,
    engine: GenerationEngine,
    output_path: str | Path,
    batch_size: int = 128,
) -> pd.DataFrame:
    output_path = Path(output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path = output_path.with_suffix(".audit.json")
    partial_path = output_path.with_name(output_path.stem + ".partial.parquet")
    progress_path = output_path.with_name(output_path.stem + ".progress.json")
    if output_path.exists() and audit_path.exists():
        audit = json.loads(audit_path.read_text())
        if int(audit.get("reactions", -1)) == len(frame) and int(audit.get("failed", -1)) == 0:
            return pd.read_parquet(output_path)
        if int(audit.get("failed", 0)) > 0:
            raise RuntimeError(
                f"Rationale generation exhausted retries for {audit['failed']} reactions; "
                "refusing to reuse an incomplete formal artifact"
            )

    prompts = [rationale_prompt(p, r) for p, r in zip(frame["product"], frame["reactants"])]
    rows: list[dict[str, str]] = []
    start_index = 0
    if partial_path.exists() and progress_path.exists():
        progress = json.loads(progress_path.read_text())
        if int(progress.get("total", -1)) != len(frame):
            raise RuntimeError("Existing rationale progress uses a different input size")
        rows = pd.read_parquet(partial_path).to_dict(orient="records")
        start_index = int(progress["next_index"])
    cache_dir = output_path.parent / "train_rationale_responses"
    reaction_ids = frame["reaction_id"].astype(str).tolist()
    for start in range(start_index, len(prompts), batch_size):
        require_free_space(output_path.parent, 8.0)
        chunk_prompts = prompts[start : start + batch_size]
        chunk_ids = reaction_ids[start : start + batch_size]
        keys = [stable_hash(reaction_id + prompt)[:24] for reaction_id, prompt in zip(chunk_ids, chunk_prompts)]
        payloads = generate_json_batch_with_retry(engine, chunk_prompts, cache_dir, keys)
        rows.extend(
            {
                "reaction_id": reaction_id,
                "rationale": json.dumps(payload, sort_keys=True) if payload is not None else "",
            }
            for reaction_id, payload in zip(chunk_ids, payloads)
        )
        partial = pd.DataFrame(rows)
        write_table(partial, partial_path)
        atomic_json(
            progress_path,
            {
                "status": "complete" if len(rows) == len(frame) else "running",
                "next_index": int(start + len(chunk_prompts)),
                "total": len(frame),
                "successful": int(partial["rationale"].ne("").sum()),
                "failed": int(partial["rationale"].eq("").sum()),
            },
        )
    result = pd.DataFrame(rows)
    if result["reaction_id"].tolist() != reaction_ids:
        raise AssertionError("Rationale checkpoint order differs from the training reactions")
    write_table(result, output_path)
    reaction_ids_hash = stable_hash("\n".join(result["reaction_id"].astype(str)))
    rationales_hash = stable_hash("\n".join(result["rationale"].astype(str)))
    atomic_json(
        audit_path,
        {
            "schema_version": 1,
            "model": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "reactions": len(result),
            "successful": int(result["rationale"].ne("").sum()),
            "failed": int(result["rationale"].eq("").sum()),
            "reaction_id_sequence_sha256": reaction_ids_hash,
            "rationale_sequence_sha256": rationales_hash,
            "output_sha256": sha256_file(output_path),
        },
    )
    failed = int(result["rationale"].eq("").sum())
    if failed:
        raise RuntimeError(
            f"Rationale generation exhausted retries for {failed} reactions; "
            "formal encoding requires complete text"
        )
    partial_path.unlink(missing_ok=True)
    return result


def encode_rationales(
    frame: pd.DataFrame,
    output_path: str | Path,
    batch_size: int = 64,
    device: str = "cuda",
) -> None:
    import torch
    from transformers import AutoModel, AutoTokenizer

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if frame["reaction_id"].astype(str).duplicated().any():
        raise ValueError("Rationale reaction IDs must be unique")
    if frame["rationale"].fillna("").astype(str).str.strip().eq("").any():
        raise ValueError("Rationale encoding refuses empty text")
    local = Path(__file__).resolve().parents[2] / "artifacts/weights/scibert"
    source = str(local) if (local / "config.json").exists() else SCIBERT_ID
    revision = None if source != SCIBERT_ID else SCIBERT_REVISION
    tokenizer = AutoTokenizer.from_pretrained(source, revision=revision)
    model = AutoModel.from_pretrained(source, revision=revision).to(device).eval()
    embeddings: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(frame), batch_size):
            texts = frame["rationale"].iloc[start : start + batch_size].fillna("").tolist()
            tokens = tokenizer(texts, padding=True, truncation=True, max_length=256, return_tensors="pt")
            tokens = {key: value.to(device) for key, value in tokens.items()}
            hidden = model(**tokens).last_hidden_state
            mask = tokens["attention_mask"].unsqueeze(-1)
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp_min(1)
            embeddings.append(pooled.cpu().numpy().astype(np.float16))
    matrix = np.concatenate(embeddings, axis=0) if embeddings else np.zeros((0, 768), np.float16)
    reaction_ids = np.asarray(frame["reaction_id"].astype(str).tolist(), dtype=np.str_)
    np.savez_compressed(output_path, reaction_id=reaction_ids, embeddings=matrix)
    rationale_values = frame["rationale"].astype(str).tolist()
    atomic_json(
        Path(output_path).with_suffix(".audit.json"),
        {
            "schema_version": 1,
            "model": SCIBERT_ID,
            "model_revision": SCIBERT_REVISION,
            "pooling": "attention-mask-weighted mean",
            "frozen": True,
            "shape": list(matrix.shape),
            "dtype": str(matrix.dtype),
            "finite": bool(np.isfinite(matrix).all()),
            "reaction_id_sequence_sha256": stable_hash("\n".join(reaction_ids.tolist())),
            "rationale_sequence_sha256": stable_hash("\n".join(rationale_values)),
            "output_sha256": sha256_file(output_path),
        },
    )

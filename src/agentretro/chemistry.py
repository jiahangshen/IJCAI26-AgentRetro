from __future__ import annotations

import contextlib
import hashlib
import io
import re
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from rdkit import Chem, DataStructs, rdBase
from rdkit.Chem import AllChem, Descriptors, GraphDescriptors, QED
from rdkit.Chem.Scaffolds import MurckoScaffold

ATOM_MAP_RE = re.compile(r":\d+(?=\])")


def split_reaction_smiles(value: str) -> tuple[str, str]:
    if ">>" in value:
        reactants, product = value.split(">>", 1)
    else:
        fields = value.split(">")
        if len(fields) != 3:
            raise ValueError(f"Invalid reaction SMILES: {value!r}")
        reactants, _, product = fields
    return reactants, product


def remove_atom_mapping(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ATOM_MAP_RE.sub("", smiles)
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)


def canonical_smiles(smiles: str, keep_mapping: bool = False) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"RDKit could not parse SMILES: {smiles!r}")
    if not keep_mapping:
        for atom in mol.GetAtoms():
            atom.SetAtomMapNum(0)
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)


def canonical_components(smiles: str, keep_mapping: bool = False) -> str:
    parts = [canonical_smiles(part, keep_mapping=keep_mapping) for part in smiles.split(".") if part]
    return ".".join(sorted(set(parts)))


def canonical_reaction(value: str) -> tuple[str, str]:
    reactants, product = split_reaction_smiles(value)
    return canonical_components(reactants), canonical_components(product)


def murcko_scaffold(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(smiles)
    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    return Chem.MolToSmiles(scaffold, canonical=True, isomericSmiles=True)


def morgan_bits(smiles: str, radius: int = 2, n_bits: int = 2048) -> np.ndarray:
    fp = morgan_fingerprint(smiles, radius=radius, n_bits=n_bits)
    out = np.zeros((n_bits,), dtype=np.uint8)
    DataStructs.ConvertToNumpyArray(fp, out)
    return out


def morgan_fingerprint(smiles: str, radius: int = 2, n_bits: int = 2048):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(smiles)
    return AllChem.GetMorganGenerator(radius=radius, fpSize=n_bits).GetFingerprint(mol)


def reaction_difference_bits(product: str, reactants: str, n_bits: int = 2048) -> np.ndarray:
    union = np.zeros((n_bits,), dtype=np.uint8)
    for smiles in reactants.split("."):
        if smiles:
            union |= morgan_bits(smiles, n_bits=n_bits)
    return np.bitwise_xor(morgan_bits(product, n_bits=n_bits), union)


def tanimoto_bits(a: np.ndarray, b: np.ndarray) -> float:
    intersection = np.bitwise_and(a, b).sum(dtype=np.float64)
    union = np.bitwise_or(a, b).sum(dtype=np.float64)
    return float(intersection / union) if union else 1.0


def molecular_complexity(smiles: str) -> dict[str, float]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(smiles)
    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    return {
        "heavy_atoms": float(mol.GetNumHeavyAtoms()),
        "bertz": float(GraphDescriptors.BertzCT(mol)),
        "scaffold_heavy_atoms": float(scaffold.GetNumHeavyAtoms()),
        "scaffold_rings": float(scaffold.GetRingInfo().NumRings()),
    }


def druglike_properties(smiles: str) -> dict[str, float]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(smiles)
    return {
        "mw": float(Descriptors.MolWt(mol)),
        "qed": float(QED.qed(mol)),
        "heavy_atoms": float(mol.GetNumHeavyAtoms()),
    }


@lru_cache(maxsize=1)
def _template_module():
    project_root = Path(__file__).resolve().parents[2]
    vendor = project_root / "third_party" / "retrocomposer"
    if str(vendor) not in sys.path:
        sys.path.insert(0, str(vendor))
    from rdchiral import template_extractor as extractor

    return extractor


def extract_retro_template(
    mapped_reaction: str,
    reaction_id: str,
    radius: int,
) -> str:
    extractor = _template_module()
    reactant_smiles, product_smiles = split_reaction_smiles(mapped_reaction)
    reactants = extractor.mols_from_smiles_list(extractor.replace_deuterated(reactant_smiles).split("."))
    products = extractor.mols_from_smiles_list(extractor.replace_deuterated(product_smiles).split("."))
    if not reactants or not products or None in reactants + products:
        raise ValueError("invalid mapped reaction")

    numpy_state = np.random.get_state()
    local_seed = int.from_bytes(
        hashlib.sha256(f"{reaction_id}|radius={radius}".encode("utf-8")).digest()[:4],
        byteorder="little",
    )
    np.random.seed(local_seed)
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            for mol in reactants + products:
                Chem.SanitizeMol(mol)
                mol.UpdatePropertyCache()
            _, changed_tags, error = extractor.get_changed_atoms(reactants, products)
            if error or not changed_tags:
                raise ValueError(f"no reaction center for {reaction_id}")
            reactant_fragments, _, _ = extractor.get_fragments_for_changed_atoms(
                reactants,
                changed_tags,
                radius=radius,
                expansion=[],
                category="reactants",
                super_general=False,
            )
            product_fragments, _, _ = extractor.get_fragments_for_changed_atoms(
                products,
                changed_tags,
                radius=0,
                expansion=extractor.expand_changed_atom_tags(changed_tags, reactant_fragments),
                category="products",
                super_general=False,
            )
            canonical = extractor.canonicalize_transform(f"{reactant_fragments}>>{product_fragments}")
    finally:
        np.random.set_state(numpy_state)
    left, right = canonical.split(">>")
    left = left[1:-1].replace(").(", ".")
    right = right[1:-1].replace(").(", ".")
    retro = f"{right}>>{left}"
    reaction = AllChem.ReactionFromSmarts(retro)
    if reaction is None or reaction.Validate()[1] != 0:
        raise ValueError(f"invalid extracted template for {reaction_id}")
    return retro


def canonical_prediction(reactants: str) -> str:
    if not reactants or not reactants.strip():
        raise ValueError("empty reactant prediction")

    with rdBase.BlockLogs():
        return canonical_components(reactants, keep_mapping=False)


def exact_match_topk(truth: str, predictions: Sequence[str], ks: Iterable[int]) -> dict[int, bool]:
    target = canonical_prediction(truth)
    unique: list[str] = []
    seen: set[str] = set()
    for prediction in predictions:
        try:
            canonical = canonical_prediction(prediction)
        except ValueError:
            continue
        if canonical not in seen:
            seen.add(canonical)
            unique.append(canonical)
    return {int(k): target in unique[: int(k)] for k in ks}

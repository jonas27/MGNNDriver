import os

import numpy as np
import pandas as pd
from graphdriver.utils import paths

get_patient_from_barcode = lambda x: "-".join(str(x).split("-")[:3])  # TCGA barcode for patient


def load(cancer: str):
    GTEX_NORMAL = True
    USE_PATIENT_NORMAL_IF_AVAILABLE = True
    # get the file names ready
    dir_name = paths.data_raw_path_pancancer()
    tumor_path = os.path.join(dir_name, "{}-rsem-fpkm-tcga-t.txt.gz")
    gtex_path = os.path.join(dir_name, "{}-rsem-fpkm-gtex.txt.gz")
    tcga_normal_path = os.path.join(dir_name, "{}-rsem-fpkm-tcga.txt.gz")

    # now, apply that function for all TCGA-project and tissue pairs

    log_fold_changes = []
    tumor_fpkm = []
    gtex_fpkm = []
    tcga_normal_fpkm = []
    mean_fold_changes = []

    gtex_tissue, tcga_project = get_tissue_pair(cancer)
    fc_gtex, tumor, gtex = compute_geneexpression_foldchange(
        tumor_path=tumor_path.format(tcga_project),
        normal_path=gtex_path.format(gtex_tissue),
        use_patients_if_possible=False,
    )

    fc_tcga, tumor, normal = compute_geneexpression_foldchange(
        tumor_path=tumor_path.format(tcga_project), normal_path=tcga_normal_path.format(tcga_project)
    )
    print(fc_tcga.shape, fc_gtex.shape)
    if GTEX_NORMAL:
        log_fold_changes.append(fc_gtex)
        mean_fold_changes.append(fc_gtex.median(axis=1))
    else:
        log_fold_changes.append(fc_tcga)
        mean_fold_changes.append(fc_tcga.median(axis=1))
    tumor_fpkm.append(tumor)
    gtex_fpkm.append(gtex)
    tcga_normal_fpkm.append(normal)

    # convert everything to one big DataFrame
    # all_foldchanges = log_fold_changes[0].join(log_fold_changes[1:])
    mean_fold_changes = pd.DataFrame(mean_fold_changes, index=[tcga_project.upper()]).T
    return mean_fold_changes


def get_a_values(col, normal):
    patients_with_normal = normal.columns.map(get_patient_from_barcode)

    if get_patient_from_barcode(col.name) in patients_with_normal:
        idx_col = patients_with_normal == get_patient_from_barcode(col.name)
        corresponding_patient_normal = normal.iloc[:, idx_col]
        average = col * corresponding_patient_normal.median(axis=1)  # * 0.5
    return average


def normalize_sample(col, normal):
    patients_with_normal = normal.columns.map(get_patient_from_barcode)
    if get_patient_from_barcode(col.name) in patients_with_normal:
        idx_col = patients_with_normal == get_patient_from_barcode(col.name)
        corresponding_patient_normal = normal.iloc[:, idx_col]
        fc = col / (corresponding_patient_normal.median(axis=1) + 1)
    else:
        print("normalizing using average normal expression")
        fc = col / (normal.median(axis=1) + 1)
    return fc


# function to get the fold changes
def compute_geneexpression_foldchange(tumor_path, normal_path, use_patients_if_possible=True):
    # read tumor and normal data
    tumor_ge = (
        pd.read_csv(tumor_path, compression="gzip", sep="\t").set_index("Hugo_Symbol").drop("Entrez_Gene_Id", axis=1)
    )
    normal_ge = (
        pd.read_csv(normal_path, compression="gzip", sep="\t").set_index("Hugo_Symbol").drop("Entrez_Gene_Id", axis=1)
    )
    assert np.all(tumor_ge.index == normal_ge.index)

    print(tumor_ge.shape, normal_ge.shape)
    # compute mean expression for tumor and normal. Then, compute log
    if use_patients_if_possible:
        original_tumor_cols = tumor_ge.columns
        patients_with_normal = normal_ge.columns.map(get_patient_from_barcode)
        tumor_ge.columns = tumor_ge.columns.map(get_patient_from_barcode)
        # now, get only tumor samples for patients that have normals, too
        tumor_normal_matched = tumor_ge.loc[:, tumor_ge.columns.isin(patients_with_normal)]
        fc = tumor_normal_matched.apply(lambda col: normalize_sample(col, normal_ge), axis=0)
        fc.columns = original_tumor_cols[tumor_ge.columns.isin(patients_with_normal)]
        print(fc.shape)
    else:
        fc = tumor_ge.divide(normal_ge.median(axis=1), axis=0)
        print(fc.shape)
    log_fc = np.log2(fc)
    log_fc = log_fc.replace([np.inf, -np.inf], np.nan).dropna(
        axis=0
    )  # remove NaN and inf (from division by 0 or 0+eta)
    print("Dropped {} genes because they contained NaNs".format(fc.shape[0] - log_fc.shape[0]))
    return log_fc, tumor_ge, normal_ge


def get_tissue_pair(cancer: str):
    tissue_pairs = [
        ("bladder", "blca"),
        ("breast", "brca"),
        ("cervix", "cesc"),
        ("uterus", "ucec"),
        ("colon", "read"),
        ("colon", "coad"),
        ("liver", "lihc"),
        ("salivary", "hnsc"),
        ("esophagus_mus", "esca"),
        ("prostate", "prad"),
        ("stomach", "stad"),
        ("thyroid", "thca"),
        ("lung", "luad"),
        ("lung", "lusc"),
        ("kidney", "kirc"),
        ("kidney", "kirp"),
    ]
    for t in tissue_pairs:
        if cancer in t:
            return t
    raise ValueError


if __name__ == "__main__":
    df = load("blca")

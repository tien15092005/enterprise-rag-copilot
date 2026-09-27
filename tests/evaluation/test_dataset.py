from pathlib import Path

from app.evaluation.datasets import dataset_sha256, load_dataset
from app.models.versioning import DocumentLanguage, VersionSelectorKind

DATASET = Path("data/evaluation/dataset.json")


def test_phase8_dataset_is_valid_bilingual_and_multi_version() -> None:
    dataset = load_dataset(DATASET)

    assert dataset.dataset_id == "enterprise-rag-bilingual-v1"
    assert len(dataset.documents) == 4
    assert sum(len(document.versions) for document in dataset.documents) == 6
    assert len(dataset.cases) == 14
    assert {case.language for case in dataset.cases} == {
        DocumentLanguage.ENGLISH,
        DocumentLanguage.VIETNAMESE,
    }
    assert {case.selector.kind for case in dataset.cases} == set(VersionSelectorKind)
    assert any("semantic_paraphrase" in case.tags for case in dataset.cases)
    assert any("exact_token" in case.tags for case in dataset.cases)
    assert sum(not case.relevant for case in dataset.cases) == 2
    assert len(dataset_sha256(DATASET)) == 64


def test_all_corpus_paths_exist() -> None:
    dataset = load_dataset(DATASET)
    assert all(
        (DATASET.parent / version.path).is_file()
        for document in dataset.documents
        for version in document.versions
    )

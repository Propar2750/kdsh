# Data Loading Task — COMPLETED ✓

## What was implemented

### 1. Data Loader Module (`pipeline/loader.py`)
- `load_csv_to_pathway()`: Loads train/test CSV, converts labels (consistent→1, contradict→0)
- `load_books()`: Loads all .txt files from Dataset/Books/, returns case-insensitive dict
- `create_story_table()`: Joins CSV + books into unified table with `novel_text` column

**Design note**: Using pandas DataFrame as Pathway table abstraction. Easy migration to actual Pathway when environment switches to Linux.

### 2. Test Suite (`tests/test_loader.py`)
7 passing tests covering:
- CSV loading + schema validation
- Label encoding (0/1 conversion)
- Book loading + content validation
- Story table creation + data integrity

### 3. Integration Test (`tests/test_integration.py`)
End-to-end test showing:
- All 4 books loaded (826K–2.6M chars each)
- All 80 training records loaded
- Labels: 51 consistent, 29 contradict
- Zero null values in critical columns

## Data verified
```
Dataset structure:
├── train.csv        (80 records, 6 columns)
├── test.csv         (not yet loaded)
└── Books/
    ├── In search of the castaways.txt      (826K chars)
    ├── The Count of Monte Cristo.txt       (2.6M chars)
    └── 2 more books loaded
```

## How to run tests

```bash
# Run all unit tests
python -m pytest tests/test_loader.py -v

# Run integration test (shows data flow)
python tests/test_integration.py

# Quick module test
python pipeline/loader.py
```

## Expected output
```
tests/test_loader.py::test_load_csv_basic PASSED                     [ 14%]
tests/test_loader.py::test_load_csv_columns PASSED                   [ 28%]
tests/test_loader.py::test_load_csv_label_encoding PASSED            [ 42%]
tests/test_loader.py::test_load_books PASSED                         [ 57%]
tests/test_loader.py::test_book_content_not_empty PASSED             [ 71%]
tests/test_loader.py::test_create_story_table PASSED                 [ 85%]
tests/test_loader.py::test_story_table_has_novel_text PASSED         [100%]

================================================ 7 passed ================================================
```

## Next tasks (tiny increments)
1. Chunk novel text + embed (with chunking strategy logged)
2. Extract atomic claims from backstory
3. Implement retriever (top-k passage lookup)
4. Verify each claim (supports/contradicts/unclear)
5. Aggregate verdicts → final label
6. Log intermediate artifacts

---
**Status**: Data ingestion ✓ | Ready for chunking/embedding step

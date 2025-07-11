@echo off
set COLLECTION_NAME=products_promotions
set CSV_PATH=docs/rag.csv

echo Creating vector store for %COLLECTION_NAME%...
python -m vector_store.main --collection-name %COLLECTION_NAME% --csv-path %CSV_PATH%

echo.
echo Done! Press any key to continue...
pause > nul

import pandas as pd


def load_csv_products(csv_path: str) -> list[dict[str, str]]:
    products_raw = pd.read_csv(csv_path).dropna()
    return products_raw.to_dict(orient="records")


def format_products_for_embedding(products: list[dict[str, str]]) -> list[str]:
    return [
        (
            f"Produto: {product['Produto']}; "
            f"Preço Original: {product['Preço Original']}; "
            f"Preço com Desconto: {product['Preço com Desconto']}; "
            f"Desconto: {product['Desconto']}; "
            f"Link: {product['Link']}"
        )
        for product in products
    ]

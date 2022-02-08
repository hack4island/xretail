from price_reader.collections.price_tag import PriceTag
from price_reader.collections.product import Product

class PriceReader:
    def __init__(self):
        self.config = None
        self.preprocessor_config = None
        self.pad_model = None
        self.ean_model = None

    def load(self):
        print("Model loaded")
        pass

    def preprocess(self, price_tag):
        print("Preprocessed price tag")
        pass

    def get_price(self, price_tag):
        print("Extracting price")
        return 0.0

    def get_ean(self, price_tag):
        print("Extracting EAN")
        return 0

    def get_product(self, price_tag):
        preprecessed_price_tag = self.preprocess(price_tag)
        product_price = self.get_price(preprecessed_price_tag)
        product_ean = self.get_ean(preprecessed_price_tag)

        product = Product(product_ean, product_price)
        return product


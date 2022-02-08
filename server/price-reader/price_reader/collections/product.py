
class Product:
    def __init__(self, ean, price=None, metadata=None):
        self.ean = ean
        self.price = price
        self.metadata = metadata

    def __str__(self):
        ean = self.ean
        price = self.price
        metadata = self.metadata
        return f"Product(EAN:{ean}) with price: {price} Metadata:{metadata}"

from dataset import load_dataset
from preprocess4MT_model import list2file


def main(product):
    FILE = "../data/ABSA-15_{}_Train_Data.xml".format(product)
    reviews = load_dataset(FILE)
    FILE = "../data/ABSA15_{}_Test.xml".format(product)
    reviews += load_dataset(FILE)

    entities = set()
    attributes = set()

    for rv in reviews:
        for stc in rv.sentences:
            for opi in stc.opinions:
                cate = opi.category
                entity, attribute = cate.split('#')
                entities.add(entity)
                attributes.add(attribute)

    list2file(entities, "../data/{}.entity".format(product))
    list2file(attributes, "../data/{}.attribute".format(product))


if __name__ == "__main__":
    main("Laptops")
    main("Restaurants")

    print "Done"

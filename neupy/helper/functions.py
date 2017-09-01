import inspect
import string

# --------------
# Define some general helper methods
# --------------
def getTrimmedAttributes(obj):
    """
    This function returns a list of attributes of the input object (class) as a list of tuples each of the form: ('field', <instance>)
    pulled from: http://stackoverflow.com/questions/9058305/getting-attributes-of-a-class
    """
    attributes = inspect.getmembers(obj, lambda a:not(inspect.isroutine(a)))
    return [a for a in attributes if not(a[0].startswith('__') and a[0].endswith('__'))]


def getFamilyAttributes(obj, family, return_labels=False):
    """"""
    attr = getTrimmedAttributes(obj)
    attributes = []
    if family is not None:
        for fam in family:
            attributes.append(attr.pop(attr.index((fam, obj.__getattribute__(fam)))))

    attrs = []
    lbls  = []
    for attribute in attributes:
        lbl, attr, = attribute
        lbls.append(lbl)
        attrs.append(attr)
    if return_labels:
        return attrs, lbls
    else:
        return attrs


def stripDigits(name):
    """"""
    pstr = string.digits
    name = name.encode('ascii','ignore')
    allNames = string.maketrans('','')
    nodig = allNames.translate(allNames, pstr)
    name = name.translate(allNames, nodig)
    return name

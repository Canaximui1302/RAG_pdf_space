import pymupdf

def pdf_r(file_name):
    doc = pymupdf.open(file_name)
    res = ""
    for i in doc:
        res += (i.get_text())
    return res
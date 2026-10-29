import pymupdf

doc = pymupdf.open("test.pdf")
print("start")
for page in doc:
    print(page.get_text())


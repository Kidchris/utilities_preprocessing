from tika import parser # pip install tika

raw = parser.from_file('Up from slavery an autobiography.pdf')
# print(raw['content'])
content = open("/home/chris/Desktop/content.txt", "w")
content.write(raw["content"])
content.close()
print("Evering is done!!!")
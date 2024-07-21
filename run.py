from infer import generate

while True:
    s = input("Enter english: ").strip()
    print("Tamil: ", end=" ")
    generate(s)
    print()
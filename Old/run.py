from infer import generate

while True:
    s = input("Enter english: ").strip()
    print("Tamil: ", end=" ")
    out = generate(s)
    with open("run.txt", "a") as f:
        f.write(f"{out}\n")
    print()
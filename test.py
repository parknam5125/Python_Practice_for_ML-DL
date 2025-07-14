def main():
    name = input("이름을 입력하세요: ")
    print(f"안녕하세요, {name}님!")

    numbers = [1, 2, 3, 4, 5]
    squared = [n**2 for n in numbers]

    print("1~5 제곱 리스트:", squared)

    total = 0
    for num in squared:
        total += num

    print("제곱 합계:", total)

    if total > 50:
        print("총합이 50보다 큽니다.")
    else:
        print("총합이 50 이하입니다.")

if __name__ == "__main__":
    main()

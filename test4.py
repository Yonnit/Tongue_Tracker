import numpy as np





def main():
    a = np.full((5, 5), 255, np.uint8)
    max = 4
    a[:, max:] = 0
    print(a)


if __name__ == '__main__':
    main()

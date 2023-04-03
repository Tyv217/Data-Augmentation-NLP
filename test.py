def main():
    import os
    import glob

    # directory containing the Python files
    dir_path = "src"

    # list of Python files in the directory
    files = glob.glob(os.path.join(dir_path, "*.py"))

    print(files)

    # initialize a variable to store the total number of lines
    total_lines = 0

    # iterate over each file and count the lines
    for root, dirs, files in os.walk(dir_path):
        # filter for only Python files
        files = [file for file in files if file.endswith(".py")]

        # iterate over each Python file and count the lines
        for file_path in files:
            with open(os.path.join(root, file_path), "r") as file:
                lines = file.readlines()
                total_lines += len(lines)

    # print the total number of lines
    print("Total number of lines:", total_lines)

if __name__ == "__main__":
    main()
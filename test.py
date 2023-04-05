import os
import subprocess

def get_libraries(directory):
    """
    Returns a list of all external Python libraries used in a directory.
    """
    libraries = set()
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".py"):
                file_path = os.path.join(root, file)
                result = subprocess.check_output(["python", "-c", "import os, sys; sys.path.append(os.path.dirname('" + file_path + "')); import pkgutil; print([name for _, name, _ in pkgutil.iter_modules()])"])
                libraries.update(eval(result))
    return list(libraries)

if __name__ == '__main__':
    libraries = get_libraries("src")
    print(libraries)
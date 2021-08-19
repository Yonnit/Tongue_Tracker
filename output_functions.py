import datetime
import ntpath
import numpy as np
import os


def save_results(user_input, tongue_lengths, process_data):
    input_file_name = path_leaf(user_input['input']).split(".")[0]
    directory_path = make_directory(user_input['output'], input_file_name)
    tf = tongue_lengths[:, 3]
    tf = tf.astype(np.bool)
    save_txt(tongue_lengths[tf], directory_path, input_file_name, 'just_maxes')
    save_txt(tongue_lengths, directory_path, input_file_name, 'tongue_lengths')
    save_process(directory_path, **process_data)
    print()
    print(f"Data Successfully saved to '{directory_path}'")
    print("Closing Program")


def make_directory(path, input_file_name):
    now = datetime.datetime.now()
    current_time = now.strftime("%Y%m%d_%H%M%S")

    directory_name = f"{input_file_name}__{current_time}"
    directory_path = os.path.join(path, directory_name)
    os.mkdir(directory_path)

    return directory_path


def save_txt(data, directory_path, input_file_name, name):
    name_and_path = f'./{directory_path}/{input_file_name}__{name}.csv'

    np.savetxt(name_and_path, data, delimiter=",")


def save_process(directory_path, **kwargs):
    for key, value in kwargs.items():
        path = os.path.join(directory_path, key)
        np.save(path, value)


# Returns the file name from a string path
def path_leaf(path):
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)



if __name__ == '__main__':
    main()
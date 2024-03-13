import os 

def get_all_files(input_dir: str):
    file_list =[os.path.join(input_dir, x) for x in os.listdir(input_dir)]
    file_list = file_list[1:]#.gitkeep
    return file_list





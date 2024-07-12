import logging
import os
import shutil

def remove_files_by_folder(folder_path):
    folder_list = list()
    for folder in os.listdir(folder_path):
        folder_list.append(folder)
    for file in folder_list:
        rm_path = os.path.join(folder_path, file)
        logging.info(f"Remove {rm_path}")
        shutil.rmtree(rm_path)

def clear_folder(folder_path):
    '''Clear all files in the specified folder'''
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            logging.error(f'Failed to delete {file_path}. Reason: {e}')
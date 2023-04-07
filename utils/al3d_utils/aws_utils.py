import os


def list_oss_dir(oss_path, client, with_info=False):
    """
    Loading files from OSS
    """
    s3_dir = fix_path(oss_path)
    files_iter = client.get_file_iterator(s3_dir)
    if with_info:
        file_list = {p: k for p, k in files_iter}
    else:
        file_list = [p for p, k in files_iter]
    return file_list

def fix_path(path_str):
    try:
        st_ = str(path_str)
        if "s3://" in st_:
            return  st_
        if "s3:/" in st_:
            st_ = "s3://" + st_.strip('s3:/')
            return st_
        else:
            st_ = "s3://" + st_
            return st_
    except:
        raise TypeError

def oss_exist(data_path, file_path, oss_data_list, refresh=False):
    if data_path is None:
        raise IndexError("No initialized path set!")
    if refresh:
        oss_data_list = list_oss_dir(data_path, with_info=False)
    pure_name = fix_path(file_path).strip("s3://")
    if pure_name in oss_data_list:
        return True
    else:
        return False
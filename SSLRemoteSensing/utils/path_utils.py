import os

def get_filename(file_path,is_suffix=True):
	file_name=file_path.replace('\\','/')
	file_name=file_name.split('/')[-1]
	if is_suffix:
		return file_name
	else:
		index=file_name.rfind('.')
		if index>0:
			return file_name[0:index]
		else:
			return file_name

def get_parent_folder(file_path,with_root=False):

	file_path=file_path.replace('\\','/')

	if os.path.isdir(file_path):
		parent_folder=file_path
	else:
		index = file_path.rfind('/')
		parent_folder=file_path[0:index]
	if not with_root:
		return get_filename(parent_folder)
	return parent_folder

if __name__=='__main__':
	path=r'G:\deep_learning\dataSet\UCMerced_LandUse\Images'
	print(get_parent_folder(path))
	print(os.listdir(path))
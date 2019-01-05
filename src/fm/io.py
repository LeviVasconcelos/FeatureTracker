import glob

def GetFilePaths(folder_path, ext):
	filepaths = []
	for path in glob.glob(folder_path + '/*.' + ext):
		filepaths.append(path)
	return filepaths, len(filepaths)


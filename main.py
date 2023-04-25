from quakenet.data_conversion import *
from quakenet.data_io import *

"""Testing data load/processing methods"""

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print("---Loading streams from MSEED/SAC files---")
    stream = load_stream("data\\mseed\\GSOK029_12-2016.mseed")
    print(stream)

    print('\n')

    print("---Loading catalogs from CSV files---")
    catalog = load_catalog("data\\catalog\\Benz_catalog.csv")
    print(catalog)

    print('\n')

    print("---Stream to numpy array---")
    stream_arr = stream2array(stream)
    print(stream_arr)

    print('\n')

    print("---Converting catalog: skipped (error)---")
    dest_path = "data\\catalog"
    src_path = "data\\catalog\\Benz_catalog.csv"
    # convert_catalog(src_path, dest_path)

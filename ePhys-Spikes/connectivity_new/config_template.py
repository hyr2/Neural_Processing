from enum import IntEnum

cellexp_path = "/home/xlruut/jiaao_workspace/stuff/CellExplorer/CellExplorer"
mda_tempdir = "/storage/SSD_2T/barrel_data_all/barrel_tempfolder/"
spk_inpdir  = "/storage/SSD_2T/barrel_data_all/barrel_data_proc/"
con_resdir  = "/storage/SSD_2T/barrel_data_all/barrel_connectivity_data_bin400us_no-axons"
# mda_reldirs = [
#     "RH-9/22-12-14",
#     "RH-9/22-12-15",
#     "RH-9/22-12-16",
#     "RH-9/22-12-17",
#     "RH-9/22-12-19",
#     "RH-9/22-12-23",
#     "RH-9/22-12-28",
#     "RH-9/23-01-04",
#     "RH-9/23-01-11",
#     "RH-9/23-01-18",
#     "RH-9/23-01-25",
#     "RH-9/23-02-01",
#     "RH-9/23-02-08",
#     #
#     # "BC8/2021-11-18",
#     # "BC8/2021-11-19",
#     # "BC8/2021-11-24",
#     # "BC8/2021-11-30-a",
#     # "BC8/2021-11-30-b",
#     # "BC8/2021-12-06",
#     # "BC8/2021-12-13",
#     # # "BC8/2021-12-19",
#     # "BC8/2022-01-09",
#     #
#     # "BC7/2021-12-06",
#     # "BC7/2021-12-07",
#     # "BC7/2021-12-09",
#     # "BC7/2022-01-07",
#     # "BC7/2022-01-21",
#     #
#     # "RH-8/2022-12-03",
#     # "RH-8/2022-12-05",
#     # "RH-8/2022-12-08",
#     # "RH-8/2022-12-13",
#     # "RH-8/2022-12-20",
#     # "RH-8/2022-12-27",
#     # "RH-8/2023-01-03",
#     # "RH-8/2023-01-10",
#     # "RH-8/2023-01-17",
#     # "RH-8/2023-01-24",
# ]
spk_reldirs = [
    "processed_data_rh9/22-12-16",
    "processed_data_rh9/22-12-17",
    "processed_data_rh9/22-12-23",
    "processed_data_rh9/22-12-28",
    "processed_data_rh9/23-01-04",
    "processed_data_rh9/23-01-11",
    "processed_data_rh9/23-01-18",
    "processed_data_rh9/23-01-25",
    "processed_data_rh9/23-02-01",
    "processed_data_rh9/23-02-08",

    "processed_data_bc7/21-12-06",
    "processed_data_bc7/21-12-07",
    "processed_data_bc7/21-12-09",
    "processed_data_bc7/21-12-12",
    "processed_data_bc7/21-12-17",
    "processed_data_bc7/21-12-24",
    "processed_data_bc7/21-12-31",
    "processed_data_bc7/22-01-07",
    "processed_data_bc7/22-01-21",

    "processed_data_rh8/22-12-01",
    "processed_data_rh8/22-12-03",
    "processed_data_rh8/22-12-05",
    "processed_data_rh8/22-12-08",
    "processed_data_rh8/22-12-13",
    "processed_data_rh8/22-12-20",
    "processed_data_rh8/22-12-27",
    "processed_data_rh8/23-01-03",
    "processed_data_rh8/23-01-10",
    "processed_data_rh8/23-01-17",
    "processed_data_rh8/23-01-24",
    "processed_data_rh8/23-01-31",

    "processed_data_rh7/22-10-17",
    "processed_data_rh7/22-10-24",
    "processed_data_rh7/22-10-27",
    "processed_data_rh7/22-10-30",
    "processed_data_rh7/22-11-04",
    "processed_data_rh7/22-11-11",
    "processed_data_rh7/22-11-21",
    "processed_data_rh7/22-11-25",
    "processed_data_rh7/22-12-02",
    "processed_data_rh7/22-12-09",
    "processed_data_rh7/22-12-16",
    "processed_data_rh7/22-12-23",

    "processed_data_rh3/22-10-02",
    "processed_data_rh3/22-10-03",
    "processed_data_rh3/22-10-04",
    "processed_data_rh3/22-10-07",
    "processed_data_rh3/22-10-12",
    "processed_data_rh3/22-10-19",
    "processed_data_rh3/22-10-26",
    "processed_data_rh3/22-11-02",
    "processed_data_rh3/22-11-30",


]
mda_reldirs = spk_reldirs

class ShankLoc(IntEnum):
    LT300 = 0
    GT300 = 1
    S2    = 2
    BAD   = 3

shank_defs = {}
shank_defs["rh3"] = [ShankLoc.LT300, ShankLoc.GT300, ShankLoc.GT300, ShankLoc.BAD]
shank_defs["rh7"] = [ShankLoc.GT300, ShankLoc.LT300, ShankLoc.LT300, ShankLoc.S2]
shank_defs["rh8"] = [ShankLoc.LT300, ShankLoc.LT300, ShankLoc.GT300, ShankLoc.S2]
shank_defs["rh9"] = [ShankLoc.LT300, ShankLoc.LT300, ShankLoc.GT300, ShankLoc.BAD]
shank_defs["bc7"] = [ShankLoc.LT300, ShankLoc.GT300, ShankLoc.S2,    ShankLoc.S2]

days_standard = [-2, -1, 2, 7, 14, 21, 28, 42, 56]
strokedays = {}
strokedays["rh3"] = "2022-10-05"
strokedays["rh7"] = "2022-10-28"
strokedays["rh8"] = "2022-12-06"
strokedays["rh9"] = "2022-12-21"
strokedays["bc7"] = "2021-12-10"

curate_params = {
    "PARAM_SPK_COUNT_TH": 300, # for rejection of sparse spikes
    "PARAM_CCG_COUNT_TH": 300,  # for rejection of sparse CCGs
    "PARAM_ACG_CONTRAST_TH": 30.0,   # for rejection of double counted units
    "PARAM_CCG_CONTRAST_TH": 30.0,    # for rejection of duplicate unit pairs
    "PARAM_TEMPLATE_SIMILARITY_TH": 0.95,
}
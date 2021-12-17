import os
import SimpleITK as sitk
import argparse
import pandas as pd
import numpy as np

train_ids = [
    "HGG/Brats18_2013_10_1/Brats18_2013_10_1",
    "HGG/Brats18_2013_11_1/Brats18_2013_11_1",
    "HGG/Brats18_2013_12_1/Brats18_2013_12_1",
    "HGG/Brats18_2013_13_1/Brats18_2013_13_1",
    "HGG/Brats18_2013_14_1/Brats18_2013_14_1",
    "HGG/Brats18_2013_17_1/Brats18_2013_17_1",
    "HGG/Brats18_2013_18_1/Brats18_2013_18_1",
    "HGG/Brats18_2013_19_1/Brats18_2013_19_1",
    "HGG/Brats18_2013_23_1/Brats18_2013_23_1",
    "HGG/Brats18_2013_25_1/Brats18_2013_25_1",
    "HGG/Brats18_2013_26_1/Brats18_2013_26_1",
    "HGG/Brats18_2013_2_1/Brats18_2013_2_1",
    "HGG/Brats18_2013_4_1/Brats18_2013_4_1",
    "HGG/Brats18_2013_5_1/Brats18_2013_5_1",
    "HGG/Brats18_CBICA_AAB_1/Brats18_CBICA_AAB_1",
    "HGG/Brats18_CBICA_AAL_1/Brats18_CBICA_AAL_1",
    "HGG/Brats18_CBICA_AAP_1/Brats18_CBICA_AAP_1",
    "HGG/Brats18_CBICA_ABE_1/Brats18_CBICA_ABE_1",
    "HGG/Brats18_CBICA_ABY_1/Brats18_CBICA_ABY_1",
    "HGG/Brats18_CBICA_ALN_1/Brats18_CBICA_ALN_1",
    "HGG/Brats18_CBICA_ALX_1/Brats18_CBICA_ALX_1",
    "HGG/Brats18_CBICA_ANG_1/Brats18_CBICA_ANG_1",
    "HGG/Brats18_CBICA_ANI_1/Brats18_CBICA_ANI_1",
    "HGG/Brats18_CBICA_ANP_1/Brats18_CBICA_ANP_1",
    "HGG/Brats18_CBICA_AOD_1/Brats18_CBICA_AOD_1",
    "HGG/Brats18_CBICA_AOH_1/Brats18_CBICA_AOH_1",
    "HGG/Brats18_CBICA_APR_1/Brats18_CBICA_APR_1",
    "HGG/Brats18_CBICA_AQG_1/Brats18_CBICA_AQG_1",
    "HGG/Brats18_CBICA_AQN_1/Brats18_CBICA_AQN_1",
    "HGG/Brats18_CBICA_AQO_1/Brats18_CBICA_AQO_1",
    "HGG/Brats18_CBICA_AQP_1/Brats18_CBICA_AQP_1",
    "HGG/Brats18_CBICA_AQQ_1/Brats18_CBICA_AQQ_1",
    "HGG/Brats18_CBICA_AQR_1/Brats18_CBICA_AQR_1",
    "HGG/Brats18_CBICA_AQU_1/Brats18_CBICA_AQU_1",
    "HGG/Brats18_CBICA_AQV_1/Brats18_CBICA_AQV_1",
    "HGG/Brats18_CBICA_AQY_1/Brats18_CBICA_AQY_1",
    "HGG/Brats18_CBICA_ARF_1/Brats18_CBICA_ARF_1",
    "HGG/Brats18_CBICA_ARW_1/Brats18_CBICA_ARW_1",
    "HGG/Brats18_CBICA_ARZ_1/Brats18_CBICA_ARZ_1",
    "HGG/Brats18_CBICA_ASA_1/Brats18_CBICA_ASA_1",
    "HGG/Brats18_CBICA_ASE_1/Brats18_CBICA_ASE_1",
    "HGG/Brats18_CBICA_ASH_1/Brats18_CBICA_ASH_1",
    "HGG/Brats18_CBICA_ASO_1/Brats18_CBICA_ASO_1",
    "HGG/Brats18_CBICA_ASU_1/Brats18_CBICA_ASU_1",
    "HGG/Brats18_CBICA_ATD_1/Brats18_CBICA_ATD_1",
    "HGG/Brats18_CBICA_ATF_1/Brats18_CBICA_ATF_1",
    "HGG/Brats18_CBICA_ATV_1/Brats18_CBICA_ATV_1",
    "HGG/Brats18_CBICA_ATX_1/Brats18_CBICA_ATX_1",
    "HGG/Brats18_CBICA_AUN_1/Brats18_CBICA_AUN_1",
    "HGG/Brats18_CBICA_AVG_1/Brats18_CBICA_AVG_1",
    "HGG/Brats18_CBICA_AWG_1/Brats18_CBICA_AWG_1",
    "HGG/Brats18_CBICA_AWI_1/Brats18_CBICA_AWI_1",
    "HGG/Brats18_CBICA_AXJ_1/Brats18_CBICA_AXJ_1",
    "HGG/Brats18_CBICA_AXL_1/Brats18_CBICA_AXL_1",
    "HGG/Brats18_CBICA_AXM_1/Brats18_CBICA_AXM_1",
    "HGG/Brats18_CBICA_AXN_1/Brats18_CBICA_AXN_1",
    "HGG/Brats18_CBICA_AXO_1/Brats18_CBICA_AXO_1",
    "HGG/Brats18_CBICA_AXQ_1/Brats18_CBICA_AXQ_1",
    "HGG/Brats18_CBICA_AYI_1/Brats18_CBICA_AYI_1",
    "HGG/Brats18_CBICA_AYU_1/Brats18_CBICA_AYU_1",
    "HGG/Brats18_CBICA_AYW_1/Brats18_CBICA_AYW_1",
    "HGG/Brats18_CBICA_AZH_1/Brats18_CBICA_AZH_1",
    "HGG/Brats18_CBICA_BFB_1/Brats18_CBICA_BFB_1",
    "HGG/Brats18_CBICA_BFP_1/Brats18_CBICA_BFP_1",
    "HGG/Brats18_CBICA_BHK_1/Brats18_CBICA_BHK_1",
    "HGG/Brats18_CBICA_BHM_1/Brats18_CBICA_BHM_1",
    "HGG/Brats18_TCIA01_150_1/Brats18_TCIA01_150_1",
    "HGG/Brats18_TCIA01_186_1/Brats18_TCIA01_186_1",
    "HGG/Brats18_TCIA01_201_1/Brats18_TCIA01_201_1",
    "HGG/Brats18_TCIA01_203_1/Brats18_TCIA01_203_1",
    "HGG/Brats18_TCIA01_221_1/Brats18_TCIA01_221_1",
    "HGG/Brats18_TCIA01_231_1/Brats18_TCIA01_231_1",
    "HGG/Brats18_TCIA01_235_1/Brats18_TCIA01_235_1",
    "HGG/Brats18_TCIA01_378_1/Brats18_TCIA01_378_1",
    "HGG/Brats18_TCIA01_401_1/Brats18_TCIA01_401_1",
    "HGG/Brats18_TCIA01_411_1/Brats18_TCIA01_411_1",
    "HGG/Brats18_TCIA01_412_1/Brats18_TCIA01_412_1",
    "HGG/Brats18_TCIA01_425_1/Brats18_TCIA01_425_1",
    "HGG/Brats18_TCIA01_448_1/Brats18_TCIA01_448_1",
    "HGG/Brats18_TCIA01_460_1/Brats18_TCIA01_460_1",
    "HGG/Brats18_TCIA01_499_1/Brats18_TCIA01_499_1",
    "HGG/Brats18_TCIA02_151_1/Brats18_TCIA02_151_1",
    "HGG/Brats18_TCIA02_168_1/Brats18_TCIA02_168_1",
    "HGG/Brats18_TCIA02_171_1/Brats18_TCIA02_171_1",
    "HGG/Brats18_TCIA02_222_1/Brats18_TCIA02_222_1",
    "HGG/Brats18_TCIA02_226_1/Brats18_TCIA02_226_1",
    "HGG/Brats18_TCIA02_274_1/Brats18_TCIA02_274_1",
    "HGG/Brats18_TCIA02_300_1/Brats18_TCIA02_300_1",
    "HGG/Brats18_TCIA02_309_1/Brats18_TCIA02_309_1",
    "HGG/Brats18_TCIA02_321_1/Brats18_TCIA02_321_1",
    "HGG/Brats18_TCIA02_322_1/Brats18_TCIA02_322_1",
    "HGG/Brats18_TCIA02_331_1/Brats18_TCIA02_331_1",
    "HGG/Brats18_TCIA02_368_1/Brats18_TCIA02_368_1",
    "HGG/Brats18_TCIA02_370_1/Brats18_TCIA02_370_1",
    "HGG/Brats18_TCIA02_374_1/Brats18_TCIA02_374_1",
    "HGG/Brats18_TCIA02_430_1/Brats18_TCIA02_430_1",
    "HGG/Brats18_TCIA02_471_1/Brats18_TCIA02_471_1",
    "HGG/Brats18_TCIA02_473_1/Brats18_TCIA02_473_1",
    "HGG/Brats18_TCIA02_491_1/Brats18_TCIA02_491_1",
    "HGG/Brats18_TCIA02_605_1/Brats18_TCIA02_605_1",
    "HGG/Brats18_TCIA02_608_1/Brats18_TCIA02_608_1",
    "HGG/Brats18_TCIA03_121_1/Brats18_TCIA03_121_1",
    "HGG/Brats18_TCIA03_133_1/Brats18_TCIA03_133_1",
    "HGG/Brats18_TCIA03_138_1/Brats18_TCIA03_138_1",
    "HGG/Brats18_TCIA03_257_1/Brats18_TCIA03_257_1",
    "HGG/Brats18_TCIA03_338_1/Brats18_TCIA03_338_1",
    "HGG/Brats18_TCIA03_375_1/Brats18_TCIA03_375_1",
    "HGG/Brats18_TCIA03_419_1/Brats18_TCIA03_419_1",
    "HGG/Brats18_TCIA03_474_1/Brats18_TCIA03_474_1",
    "HGG/Brats18_TCIA04_149_1/Brats18_TCIA04_149_1",
    "HGG/Brats18_TCIA04_192_1/Brats18_TCIA04_192_1",
    "HGG/Brats18_TCIA04_437_1/Brats18_TCIA04_437_1",
    "HGG/Brats18_TCIA04_479_1/Brats18_TCIA04_479_1",
    "HGG/Brats18_TCIA05_396_1/Brats18_TCIA05_396_1",
    "HGG/Brats18_TCIA05_444_1/Brats18_TCIA05_444_1",
    "HGG/Brats18_TCIA06_184_1/Brats18_TCIA06_184_1",
    "HGG/Brats18_TCIA06_247_1/Brats18_TCIA06_247_1",
    "HGG/Brats18_TCIA06_372_1/Brats18_TCIA06_372_1",
    "HGG/Brats18_TCIA06_409_1/Brats18_TCIA06_409_1",
    "HGG/Brats18_TCIA08_105_1/Brats18_TCIA08_105_1",
    "HGG/Brats18_TCIA08_113_1/Brats18_TCIA08_113_1",
    "HGG/Brats18_TCIA08_162_1/Brats18_TCIA08_162_1",
    "HGG/Brats18_TCIA08_167_1/Brats18_TCIA08_167_1",
    "HGG/Brats18_TCIA08_205_1/Brats18_TCIA08_205_1",
    "HGG/Brats18_TCIA08_218_1/Brats18_TCIA08_218_1",
    "HGG/Brats18_TCIA08_242_1/Brats18_TCIA08_242_1",
    "HGG/Brats18_TCIA08_280_1/Brats18_TCIA08_280_1",
    "HGG/Brats18_TCIA08_319_1/Brats18_TCIA08_319_1",
    "LGG/Brats18_2013_0_1/Brats18_2013_0_1",
    "LGG/Brats18_2013_15_1/Brats18_2013_15_1",
    "LGG/Brats18_2013_16_1/Brats18_2013_16_1",
    "LGG/Brats18_2013_28_1/Brats18_2013_28_1",
    "LGG/Brats18_2013_6_1/Brats18_2013_6_1",
    "LGG/Brats18_2013_9_1/Brats18_2013_9_1",
    "LGG/Brats18_TCIA09_254_1/Brats18_TCIA09_254_1",
    "LGG/Brats18_TCIA09_255_1/Brats18_TCIA09_255_1",
    "LGG/Brats18_TCIA09_312_1/Brats18_TCIA09_312_1",
    "LGG/Brats18_TCIA09_402_1/Brats18_TCIA09_402_1",
    "LGG/Brats18_TCIA09_428_1/Brats18_TCIA09_428_1",
    "LGG/Brats18_TCIA09_451_1/Brats18_TCIA09_451_1",
    "LGG/Brats18_TCIA09_462_1/Brats18_TCIA09_462_1",
    "LGG/Brats18_TCIA09_620_1/Brats18_TCIA09_620_1",
    "LGG/Brats18_TCIA10_109_1/Brats18_TCIA10_109_1",
    "LGG/Brats18_TCIA10_266_1/Brats18_TCIA10_266_1",
    "LGG/Brats18_TCIA10_282_1/Brats18_TCIA10_282_1",
    "LGG/Brats18_TCIA10_307_1/Brats18_TCIA10_307_1",
    "LGG/Brats18_TCIA10_310_1/Brats18_TCIA10_310_1",
    "LGG/Brats18_TCIA10_330_1/Brats18_TCIA10_330_1",
    "LGG/Brats18_TCIA10_351_1/Brats18_TCIA10_351_1",
    "LGG/Brats18_TCIA10_393_1/Brats18_TCIA10_393_1",
    "LGG/Brats18_TCIA10_410_1/Brats18_TCIA10_410_1",
    "LGG/Brats18_TCIA10_413_1/Brats18_TCIA10_413_1",
    "LGG/Brats18_TCIA10_442_1/Brats18_TCIA10_442_1",
    "LGG/Brats18_TCIA10_449_1/Brats18_TCIA10_449_1",
    "LGG/Brats18_TCIA10_490_1/Brats18_TCIA10_490_1",
    "LGG/Brats18_TCIA10_625_1/Brats18_TCIA10_625_1",
    "LGG/Brats18_TCIA10_628_1/Brats18_TCIA10_628_1",
    "LGG/Brats18_TCIA10_632_1/Brats18_TCIA10_632_1",
    "LGG/Brats18_TCIA10_637_1/Brats18_TCIA10_637_1",
    "LGG/Brats18_TCIA10_639_1/Brats18_TCIA10_639_1",
    "LGG/Brats18_TCIA10_640_1/Brats18_TCIA10_640_1",
    "LGG/Brats18_TCIA10_644_1/Brats18_TCIA10_644_1",
    "LGG/Brats18_TCIA12_298_1/Brats18_TCIA12_298_1",
    "LGG/Brats18_TCIA12_466_1/Brats18_TCIA12_466_1",
    "LGG/Brats18_TCIA12_470_1/Brats18_TCIA12_470_1",
    "LGG/Brats18_TCIA12_480_1/Brats18_TCIA12_480_1",
    "LGG/Brats18_TCIA13_615_1/Brats18_TCIA13_615_1",
    "LGG/Brats18_TCIA13_618_1/Brats18_TCIA13_618_1",
    "LGG/Brats18_TCIA13_623_1/Brats18_TCIA13_623_1",
    "LGG/Brats18_TCIA13_633_1/Brats18_TCIA13_633_1",
    "LGG/Brats18_TCIA13_645_1/Brats18_TCIA13_645_1",
]

valid_ids = [
    "HGG/Brats18_2013_21_1/Brats18_2013_21_1",
    "HGG/Brats18_2013_22_1/Brats18_2013_22_1",
    "HGG/Brats18_2013_3_1/Brats18_2013_3_1",
    "HGG/Brats18_CBICA_AAG_1/Brats18_CBICA_AAG_1",
    "HGG/Brats18_CBICA_ABM_1/Brats18_CBICA_ABM_1",
    "HGG/Brats18_CBICA_AME_1/Brats18_CBICA_AME_1",
    "HGG/Brats18_CBICA_AOP_1/Brats18_CBICA_AOP_1",
    "HGG/Brats18_CBICA_AOZ_1/Brats18_CBICA_AOZ_1",
    "HGG/Brats18_CBICA_AQA_1/Brats18_CBICA_AQA_1",
    "HGG/Brats18_CBICA_AQT_1/Brats18_CBICA_AQT_1",
    "HGG/Brats18_CBICA_AQZ_1/Brats18_CBICA_AQZ_1",
    "HGG/Brats18_CBICA_ASY_1/Brats18_CBICA_ASY_1",
    "HGG/Brats18_CBICA_ATB_1/Brats18_CBICA_ATB_1",
    "HGG/Brats18_CBICA_AUQ_1/Brats18_CBICA_AUQ_1",
    "HGG/Brats18_CBICA_AUR_1/Brats18_CBICA_AUR_1",
    "HGG/Brats18_CBICA_AVJ_1/Brats18_CBICA_AVJ_1",
    "HGG/Brats18_CBICA_AVV_1/Brats18_CBICA_AVV_1",
    "HGG/Brats18_CBICA_AXW_1/Brats18_CBICA_AXW_1",
    "HGG/Brats18_CBICA_AYA_1/Brats18_CBICA_AYA_1",
    "HGG/Brats18_CBICA_BHB_1/Brats18_CBICA_BHB_1",
    "HGG/Brats18_TCIA01_390_1/Brats18_TCIA01_390_1",
    "HGG/Brats18_TCIA01_429_1/Brats18_TCIA01_429_1",
    "HGG/Brats18_TCIA02_118_1/Brats18_TCIA02_118_1",
    "HGG/Brats18_TCIA02_179_1/Brats18_TCIA02_179_1",
    "HGG/Brats18_TCIA02_283_1/Brats18_TCIA02_283_1",
    "HGG/Brats18_TCIA02_314_1/Brats18_TCIA02_314_1",
    "HGG/Brats18_TCIA02_377_1/Brats18_TCIA02_377_1",
    "HGG/Brats18_TCIA02_394_1/Brats18_TCIA02_394_1",
    "HGG/Brats18_TCIA02_606_1/Brats18_TCIA02_606_1",
    "HGG/Brats18_TCIA02_607_1/Brats18_TCIA02_607_1",
    "HGG/Brats18_TCIA03_265_1/Brats18_TCIA03_265_1",
    "HGG/Brats18_TCIA03_498_1/Brats18_TCIA03_498_1",
    "HGG/Brats18_TCIA04_343_1/Brats18_TCIA04_343_1",
    "HGG/Brats18_TCIA04_361_1/Brats18_TCIA04_361_1",
    "HGG/Brats18_TCIA05_478_1/Brats18_TCIA05_478_1",
    "HGG/Brats18_TCIA06_603_1/Brats18_TCIA06_603_1",
    "HGG/Brats18_TCIA08_234_1/Brats18_TCIA08_234_1",
    "HGG/Brats18_TCIA08_406_1/Brats18_TCIA08_406_1",
    "HGG/Brats18_TCIA08_436_1/Brats18_TCIA08_436_1",
    "HGG/Brats18_TCIA08_469_1/Brats18_TCIA08_469_1",
    "LGG/Brats18_2013_24_1/Brats18_2013_24_1",
    "LGG/Brats18_2013_8_1/Brats18_2013_8_1",
    "LGG/Brats18_TCIA09_141_1/Brats18_TCIA09_141_1",
    "LGG/Brats18_TCIA10_103_1/Brats18_TCIA10_103_1",
    "LGG/Brats18_TCIA10_175_1/Brats18_TCIA10_175_1",
    "LGG/Brats18_TCIA10_261_1/Brats18_TCIA10_261_1",
    "LGG/Brats18_TCIA10_276_1/Brats18_TCIA10_276_1",
    "LGG/Brats18_TCIA10_325_1/Brats18_TCIA10_325_1",
    "LGG/Brats18_TCIA10_387_1/Brats18_TCIA10_387_1",
    "LGG/Brats18_TCIA10_420_1/Brats18_TCIA10_420_1",
    "LGG/Brats18_TCIA12_101_1/Brats18_TCIA12_101_1",
    "LGG/Brats18_TCIA12_249_1/Brats18_TCIA12_249_1",
    "LGG/Brats18_TCIA13_624_1/Brats18_TCIA13_624_1",
    "LGG/Brats18_TCIA13_630_1/Brats18_TCIA13_630_1",
    "LGG/Brats18_TCIA13_634_1/Brats18_TCIA13_634_1",
    "LGG/Brats18_TCIA13_642_1/Brats18_TCIA13_642_1",
    "LGG/Brats18_TCIA13_650_1/Brats18_TCIA13_650_1",
]

test_ids = [
    "HGG/Brats18_2013_20_1/Brats18_2013_20_1",
    "HGG/Brats18_2013_27_1/Brats18_2013_27_1",
    "HGG/Brats18_2013_7_1/Brats18_2013_7_1",
    "HGG/Brats18_CBICA_ABB_1/Brats18_CBICA_ABB_1",
    "HGG/Brats18_CBICA_ABN_1/Brats18_CBICA_ABN_1",
    "HGG/Brats18_CBICA_ABO_1/Brats18_CBICA_ABO_1",
    "HGG/Brats18_CBICA_ALU_1/Brats18_CBICA_ALU_1",
    "HGG/Brats18_CBICA_AMH_1/Brats18_CBICA_AMH_1",
    "HGG/Brats18_CBICA_ANZ_1/Brats18_CBICA_ANZ_1",
    "HGG/Brats18_CBICA_AOO_1/Brats18_CBICA_AOO_1",
    "HGG/Brats18_CBICA_APY_1/Brats18_CBICA_APY_1",
    "HGG/Brats18_CBICA_APZ_1/Brats18_CBICA_APZ_1",
    "HGG/Brats18_CBICA_AQD_1/Brats18_CBICA_AQD_1",
    "HGG/Brats18_CBICA_AQJ_1/Brats18_CBICA_AQJ_1",
    "HGG/Brats18_CBICA_ASG_1/Brats18_CBICA_ASG_1",
    "HGG/Brats18_CBICA_ASK_1/Brats18_CBICA_ASK_1",
    "HGG/Brats18_CBICA_ASN_1/Brats18_CBICA_ASN_1",
    "HGG/Brats18_CBICA_ASV_1/Brats18_CBICA_ASV_1",
    "HGG/Brats18_CBICA_ASW_1/Brats18_CBICA_ASW_1",
    "HGG/Brats18_CBICA_ATP_1/Brats18_CBICA_ATP_1",
    "HGG/Brats18_CBICA_AWH_1/Brats18_CBICA_AWH_1",
    "HGG/Brats18_CBICA_AZD_1/Brats18_CBICA_AZD_1",
    "HGG/Brats18_TCIA01_131_1/Brats18_TCIA01_131_1",
    "HGG/Brats18_TCIA01_147_1/Brats18_TCIA01_147_1",
    "HGG/Brats18_TCIA01_180_1/Brats18_TCIA01_180_1",
    "HGG/Brats18_TCIA01_190_1/Brats18_TCIA01_190_1",
    "HGG/Brats18_TCIA01_335_1/Brats18_TCIA01_335_1",
    "HGG/Brats18_TCIA02_117_1/Brats18_TCIA02_117_1",
    "HGG/Brats18_TCIA02_135_1/Brats18_TCIA02_135_1",
    "HGG/Brats18_TCIA02_198_1/Brats18_TCIA02_198_1",
    "HGG/Brats18_TCIA02_208_1/Brats18_TCIA02_208_1",
    "HGG/Brats18_TCIA02_290_1/Brats18_TCIA02_290_1",
    "HGG/Brats18_TCIA02_455_1/Brats18_TCIA02_455_1",
    "HGG/Brats18_TCIA03_199_1/Brats18_TCIA03_199_1",
    "HGG/Brats18_TCIA03_296_1/Brats18_TCIA03_296_1",
    "HGG/Brats18_TCIA04_111_1/Brats18_TCIA04_111_1",
    "HGG/Brats18_TCIA04_328_1/Brats18_TCIA04_328_1",
    "HGG/Brats18_TCIA05_277_1/Brats18_TCIA05_277_1",
    "HGG/Brats18_TCIA06_165_1/Brats18_TCIA06_165_1",
    "HGG/Brats18_TCIA06_211_1/Brats18_TCIA06_211_1",
    "HGG/Brats18_TCIA06_332_1/Brats18_TCIA06_332_1",
    "HGG/Brats18_TCIA08_278_1/Brats18_TCIA08_278_1",
    "LGG/Brats18_2013_1_1/Brats18_2013_1_1",
    "LGG/Brats18_2013_29_1/Brats18_2013_29_1",
    "LGG/Brats18_TCIA09_177_1/Brats18_TCIA09_177_1",
    "LGG/Brats18_TCIA09_493_1/Brats18_TCIA09_493_1",
    "LGG/Brats18_TCIA10_130_1/Brats18_TCIA10_130_1",
    "LGG/Brats18_TCIA10_152_1/Brats18_TCIA10_152_1",
    "LGG/Brats18_TCIA10_202_1/Brats18_TCIA10_202_1",
    "LGG/Brats18_TCIA10_241_1/Brats18_TCIA10_241_1",
    "LGG/Brats18_TCIA10_299_1/Brats18_TCIA10_299_1",
    "LGG/Brats18_TCIA10_346_1/Brats18_TCIA10_346_1",
    "LGG/Brats18_TCIA10_408_1/Brats18_TCIA10_408_1",
    "LGG/Brats18_TCIA10_629_1/Brats18_TCIA10_629_1",
    "LGG/Brats18_TCIA13_621_1/Brats18_TCIA13_621_1",
    "LGG/Brats18_TCIA13_653_1/Brats18_TCIA13_653_1",
    "LGG/Brats18_TCIA13_654_1/Brats18_TCIA13_654_1",
]


def get_brain_mask(t1):
    brain_mask = sitk.GetImageFromArray(
        (sitk.GetArrayFromImage(t1) > 0).astype(np.uint8)
    )
    brain_mask.CopyInformation(t1)
    brain_mask = sitk.Cast(brain_mask, sitk.sitkUInt8)
    return brain_mask


def z_score_normalisation(
    channel, brain_mask, cutoff_percentiles=(5.0, 95.0), cutoff_below_mean=True
):
    low, high = np.percentile(channel[brain_mask.astype(np.bool)], cutoff_percentiles)
    norm_mask = np.logical_and(
        brain_mask, np.logical_and(channel > low, channel < high)
    )
    if cutoff_below_mean:
        norm_mask = np.logical_and(norm_mask, channel > np.mean(channel))
    masked_channel = channel[norm_mask]
    normalised_channel = (channel - np.mean(masked_channel)) / np.std(masked_channel)
    return normalised_channel


def fix_segmentation_labels(seg):
    array = sitk.GetArrayFromImage(seg)
    array[array == 4] = 3
    new_seg = sitk.GetImageFromArray(array)
    new_seg.CopyInformation(seg)
    return new_seg


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir", required=True, type=str, help="Path to input directory."
    )
    parser.add_argument(
        "--output-dir", required=True, type=str, help="Path to output directory."
    )

    parse_args, unknown = parser.parse_known_args()
    output_dataframe = pd.DataFrame()
    for subdir_0 in ["HGG", "LGG"]:
        for subdir_1 in os.listdir(os.path.join(parse_args.input_dir, subdir_0)):
            id_ = os.path.join(subdir_0, subdir_1) + "/" + subdir_1
            print(id_)
            seg = fix_segmentation_labels(
                sitk.ReadImage(os.path.join(parse_args.input_dir, id_) + "_seg.nii.gz")
            )
            output_path = os.path.join(parse_args.output_dir, id_) + f"_seg.nii.gz"
            output_dataframe.loc[id_, "seg"] = output_path
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            sitk.WriteImage(seg, output_path)

            t1 = sitk.ReadImage(os.path.join(parse_args.input_dir, id_) + "_t1.nii.gz")
            brain_mask = get_brain_mask(t1)
            output_path = (
                os.path.join(parse_args.output_dir, id_) + f"_brain_mask.nii.gz"
            )
            output_dataframe.loc[id_, "sampling_mask"] = output_path
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            sitk.WriteImage(brain_mask, output_path)

            for suffix in ["flair", "t1", "t1ce", "t2"]:
                channel = sitk.ReadImage(
                    os.path.join(parse_args.input_dir, id_) + f"_{suffix:s}.nii.gz"
                )
                channel_array = sitk.GetArrayFromImage(channel)
                normalised_channel_array = z_score_normalisation(
                    channel_array, sitk.GetArrayFromImage(brain_mask)
                )
                normalised_channel = sitk.GetImageFromArray(normalised_channel_array)
                normalised_channel.CopyInformation(channel)
                output_path = (
                    os.path.join(parse_args.output_dir, id_) + f"_{suffix:s}.nii.gz"
                )
                output_dataframe.loc[id_, suffix] = output_path
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                sitk.WriteImage(normalised_channel, output_path)
    output_dataframe.index.name = "id"
    os.makedirs("assets/BraTS2018_data", exist_ok=True)
    train_index = output_dataframe.loc[train_ids]
    train_index.to_csv("assets/BraTS2018_data/data_index_train.csv")
    valid_index = output_dataframe.loc[valid_ids]
    valid_index.to_csv("assets/BraTS2018_data/data_index_valid.csv")
    test_index = output_dataframe.loc[test_ids]
    test_index.to_csv("assets/BraTS2018_data/data_index_test.csv")

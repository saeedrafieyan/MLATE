BIOMATERIAL_OPTIONS = [
    'Alginate (%w/v)',
    'PVA-HA (%w/v)',
    'CaSO4 (%w/v)',
    'Na2HPO4 (%w/v)',
    'Gelatin (%w/v)',
    'GelMA (%w/v)',
    'laponite (%w/v)',
    'graphene oxide (%w/v)',
    'hydroxyapatite (%w/v)',
    'Hyaluronic_Acid (%w/v)',
    'hyaluronan methacrylate (%w/v)',
    'NorHA (%w/v)',
    'Fibroin/Fibrinogen (%w/v)',
    'Pluronic P-123 (%w/v)',
    'Collagen (%w/v)',
    'Chitosan (%w/v)',
    'CS-AEMA (%w/v)',
    'RGD (mM)',
    'TCP (%w/v)',
    'Gellan (%w/v)',
    'bioactive glass (%w/v)',
    'Nano/Methylcellulose (%w/v)',
    'PEGTA (%w/v)',
    'PEGMA (%w/v)',
    'PEGDA (%w/v)',
    'Agarose (%w/v)',
    'hyaluronic acid+ Ph moieties (%w/v)',
    'matrigel (%w/v)',
    'CaCl2 (mM)',
    'NaCl (mM)',
    'BaCl2 (mM)',
    'SrCl2 (mM)',
    'CaCO3 (mM)',
    'Genipin (%w/v)',
    'PVA (%wt)',
    'trans-glutaminase (%w/v)',
    'alginate lyase (U/ml)',
    'D-glucose (%w/v)',
    'PLGA (%w/v)',
    'vascular tissue-derived dECM (%w/v)',
    'PEG-8-SH (mM)',
    'Alginate dialdehyde (%w/v)',
    'Alginate sulfate (%w/v)',
    'RGD-modified alginate (%w/v)',
    'poly(N-isopropylacrylamide) grafted hyaluronan (%w/v)',
    'chondroitin sulfate methacrylate (%w/v)',
    'PCL (%w/v)',
    'alginate methacrylate (%w/v)',
    'HRP (U/ml)',
    'Pluronic F127 / Lutrol F127 (%w/v)',
    'Irgacure 2959 (%w/v)',
    'Eosin Y (%w/v)',
    'Ruthenium (mM)',
    'sodium persulfate (SPS) (mM)',
    'HEPES (mM)',
    'LAP (%w/v)',
    'glutaraldehyde (%w/v)',
    'PBS (M)',
    'glycerol (%w/v)',
    'cECM (%w/v)',
    'gel-fu (%w/v)',
    'Rose Bengal (%w/v)',
    'Vitamin B2 (%w/v)',
    'VEGF (%w/v)',
    'Polypyrrole:PSS (%w/v)',
    'borate bioactive glass (%w/v)',
    'astaxanthin (%w/v)',
    'PRP (%v/v)',
    'methacrylated collagen (%w/v)',
    'α-Toc (µM)',
    'ascorbic acid (mM)',
    'Liver dECM (%w/v)',
    'galactosylated alginate (%w/v)',
    'SC-PEG (%w/v)',
    'SFMA-L (%w/v)',
    'SFMA-M (%w/v)',
    'SFMA-H (%w/v)',
    'KdECMMA (%w/v)',
    'BA silk fibroin (%w/v)',
    'Carrageenan (%v)',
    'Carbopol ETD 2020 NF (%w/v)',
    'Carbopol Ultrez 10 NF (%w/v)',
    'Carbopol NF-980 (%w/v)',
    'FBS (%v/v)',
    'MeTro (%w/v)',
    'Triethanolamine (%v/v)',
    'PEG-Fibrinogen (%w/v)',
    'polyethylene glycol dimethacrylate (%w/v)',
    'aprotinin (µg/ml)',
    'gold nanorod (mg/mL)',
    'egg white (w/v)',
    '1-Vinyl-2-pyrrolidone (v/v)',
    'carboxyl functionalized carbon nanotubes (%w/v)',
    'polyHIPE (%w/v)',
    'β-D galactose (mM)',
    'hydrogen peroxide (H2O2) (%v/v)',
    'lactic acid (v/v)',
    'NorCol (%w/v)',
    'DTT (%w/v)',
    'ammonium persulfate (mM)',
    'diTyr-RGD (mM)',
    'PHEG-Tyr (%w/v)',
    'MMP2-degradable peptide (%w/v)',
    'KdECM (%w/v)',
    'EDC (mg)',
    'NHS (mg)',
    'VA086 (%w/v)',
    'PGS (%w/v)',
    'thiolated HA (%w/v)',
    'boron nitride nanotubes (%w/v)',
    'PEDOT:PSS (ul)',
    'KCl (mM)',
    'skeletal muscle ECM methacrylate (%w/v)',
    'PEO (%w/v)',
    'Carbon dots (mg/ml)',
    'Laminin (ug/ml)',
    'DF-PEG (%w/v)',
    'omentum ECM (%w/v)',
    'thrombin (unit/ml)',
    'Carbon nanotube (CNT) (w/v)',
    'Phytagel (%v)',
    'Laponite-XLG (%w/w)',
    'sodium carboxymethyl cellulose (mg)',
    'lysozyme amyloid nanofibrils:gold nanoparticles (mg)',
    'Lysozyme amyloid nanofibrils (mg)',
    'rGO (mg/ml)',
    'Methacrylated gellan gum (%w/v)',
    'Acetylsalicylic Acid (%w/w)',
    'PVA methacrylate (%w/v)',
    'MXene (mg/ml)',
]

BIOMATERIAL_RANGES = {
    "Alginate (%w/v)": {
        "min": 0.25,
        "max": 20.0,
        "median": 4.0,
        "n": 1138
    },
    "PVA-HA (%w/v)": {
        "min": 0.3,
        "max": 2.5,
        "median": 2.0,
        "n": 15
    },
    "CaSO4 (%w/v)": {
        "min": 0.03,
        "max": 25.71,
        "median": 1.0,
        "n": 32
    },
    "Na2HPO4 (%w/v)": {
        "min": 0.12,
        "max": 0.15,
        "median": 0.15,
        "n": 9
    },
    "Gelatin (%w/v)": {
        "min": 0.5,
        "max": 80.0,
        "median": 5.0,
        "n": 797
    },
    "GelMA (%w/v)": {
        "min": 1.0,
        "max": 30.0,
        "median": 7.0,
        "n": 880
    },
    "laponite (%w/v)": {
        "min": 0.05,
        "max": 2.3,
        "median": 2.3,
        "n": 96
    },
    "graphene oxide (%w/v)": {
        "min": 0.001,
        "max": 1.5,
        "median": 0.5,
        "n": 43
    },
    "hydroxyapatite (%w/v)": {
        "min": 2.0,
        "max": 70.0,
        "median": 70.0,
        "n": 43
    },
    "Hyaluronic_Acid (%w/v)": {
        "min": 0.2,
        "max": 2.0,
        "median": 0.3,
        "n": 62
    },
    "hyaluronan methacrylate (%w/v)": {
        "min": 0.1,
        "max": 6.0,
        "median": 2.0,
        "n": 151
    },
    "NorHA (%w/v)": {
        "min": 2.0,
        "max": 2.0,
        "median": 2.0,
        "n": 4
    },
    "Fibroin/Fibrinogen (%w/v)": {
        "min": 0.009375,
        "max": 25.0,
        "median": 2.0,
        "n": 207
    },
    "Pluronic P-123 (%w/v)": {
        "min": 40.0,
        "max": 60.0,
        "median": 50.0,
        "n": 12
    },
    "Collagen (%w/v)": {
        "min": 0.01,
        "max": 7.8,
        "median": 2.0,
        "n": 85
    },
    "Chitosan (%w/v)": {
        "min": 1.0,
        "max": 62.0,
        "median": 2.0,
        "n": 74
    },
    "CS-AEMA (%w/v)": {
        "min": 4.0,
        "max": 4.0,
        "median": 4.0,
        "n": 8
    },
    "RGD (mM)": {
        "min": 3.0,
        "max": 3.0,
        "median": 3.0,
        "n": 1
    },
    "TCP (%w/v)": {
        "min": 0.5,
        "max": 3.0,
        "median": 0.5,
        "n": 9
    },
    "Gellan (%w/v)": {
        "min": 0.5,
        "max": 150.0,
        "median": 150.0,
        "n": 54
    },
    "bioactive glass (%w/v)": {
        "min": 1.0,
        "max": 50.0,
        "median": 1.0,
        "n": 7
    },
    "Nano/Methylcellulose (%w/v)": {
        "min": 0.25,
        "max": 80.0,
        "median": 4.0,
        "n": 157
    },
    "PEGTA (%w/v)": {
        "min": 1.0,
        "max": 3.0,
        "median": 2.0,
        "n": 33
    },
    "PEGMA (%w/v)": {
        "min": 12.5,
        "max": 100.0,
        "median": 17.5,
        "n": 9
    },
    "PEGDA (%w/v)": {
        "min": 1.0,
        "max": 20.0,
        "median": 8.25,
        "n": 119
    },
    "Agarose (%w/v)": {
        "min": 0.5,
        "max": 60.0,
        "median": 24.0,
        "n": 51
    },
    "hyaluronic acid+ Ph moieties (%w/v)": {
        "min": 0.1,
        "max": 1.5,
        "median": 0.8,
        "n": 8
    },
    "matrigel (%w/v)": {
        "min": 5.0,
        "max": 50.0,
        "median": 10.0,
        "n": 19
    },
    "CaCl2 (mM)": {
        "min": 0.001,
        "max": 1500.0,
        "median": 100.0,
        "n": 1057
    },
    "NaCl (mM)": {
        "min": 0.72,
        "max": 350.0,
        "median": 145.0,
        "n": 23
    },
    "BaCl2 (mM)": {
        "min": 55.0,
        "max": 60.0,
        "median": 55.0,
        "n": 12
    },
    "SrCl2 (mM)": {
        "min": 20.0,
        "max": 70.0,
        "median": 20.0,
        "n": 6
    },
    "CaCO3 (mM)": {
        "min": 15.88,
        "max": 25.71,
        "median": 18.0,
        "n": 9
    },
    "Genipin (%w/v)": {
        "min": 0.025,
        "max": 1.0,
        "median": 0.025,
        "n": 18
    },
    "PVA (%wt)": {
        "min": 3.0,
        "max": 15.0,
        "median": 15.0,
        "n": 50
    },
    "trans-glutaminase (%w/v)": {
        "min": 0.04,
        "max": 6.0,
        "median": 1.0,
        "n": 63
    },
    "alginate lyase (U/ml)": {
        "min": 0.5,
        "max": 500.0,
        "median": 5.0,
        "n": 7
    },
    "D-glucose (%w/v)": {
        "min": 4.4,
        "max": 4.4,
        "median": 4.4,
        "n": 5
    },
    "PLGA (%w/v)": {
        "min": 100.0,
        "max": 100.0,
        "median": 100.0,
        "n": 4
    },
    "vascular tissue-derived dECM (%w/v)": {
        "min": 1.0,
        "max": 3.0,
        "median": 3.0,
        "n": 7
    },
    "PEG-8-SH (mM)": {
        "min": 2.25,
        "max": 8.0,
        "median": 2.25,
        "n": 10
    },
    "Alginate dialdehyde (%w/v)": {
        "min": 2.0,
        "max": 7.5,
        "median": 3.75,
        "n": 48
    },
    "Alginate sulfate (%w/v)": {
        "min": 1.0,
        "max": 1.0,
        "median": 1.0,
        "n": 26
    },
    "RGD-modified alginate (%w/v)": {
        "min": 1.0,
        "max": 1.0,
        "median": 1.0,
        "n": 2
    },
    "poly(N-isopropylacrylamide) grafted hyaluronan (%w/v)": {
        "min": 15.0,
        "max": 15.0,
        "median": 15.0,
        "n": 3
    },
    "chondroitin sulfate methacrylate (%w/v)": {
        "min": 5.0,
        "max": 5.0,
        "median": 5.0,
        "n": 1
    },
    "PCL (%w/v)": {
        "min": 1.0,
        "max": 100.0,
        "median": 8.0,
        "n": 34
    },
    "alginate methacrylate (%w/v)": {
        "min": 1.0,
        "max": 3.0,
        "median": 3.0,
        "n": 24
    },
    "HRP (U/ml)": {
        "min": 5.0,
        "max": 100.0,
        "median": 15.0,
        "n": 68
    },
    "Pluronic F127 / Lutrol F127 (%w/v)": {
        "min": 3.0,
        "max": 100.0,
        "median": 6.0,
        "n": 83
    },
    "Irgacure 2959 (%w/v)": {
        "min": 0.01,
        "max": 2.0,
        "median": 0.25,
        "n": 530
    },
    "Eosin Y (%w/v)": {
        "min": 0.5,
        "max": 100.0,
        "median": 0.5,
        "n": 32
    },
    "Ruthenium (mM)": {
        "min": 0.254,
        "max": 1.0,
        "median": 0.5,
        "n": 28
    },
    "sodium persulfate (SPS) (mM)": {
        "min": 2.52,
        "max": 10.0,
        "median": 5.0,
        "n": 28
    },
    "HEPES (mM)": {
        "min": 10.0,
        "max": 25.0,
        "median": 10.0,
        "n": 22
    },
    "LAP (%w/v)": {
        "min": 0.01,
        "max": 4.46,
        "median": 0.4,
        "n": 310
    },
    "glutaraldehyde (%w/v)": {
        "min": 0.125,
        "max": 0.4,
        "median": 0.25,
        "n": 62
    },
    "PBS (M)": {
        "min": 0.082,
        "max": 0.328,
        "median": 0.165,
        "n": 3
    },
    "glycerol (%w/v)": {
        "min": 10.0,
        "max": 10.0,
        "median": 10.0,
        "n": 21
    },
    "cECM (%w/v)": {
        "min": 0.1,
        "max": 20.0,
        "median": 4.6,
        "n": 118
    },
    "gel-fu (%w/v)": {
        "min": 10.0,
        "max": 155.0,
        "median": 100.0,
        "n": 15
    },
    "Rose Bengal (%w/v)": {
        "min": 1.0,
        "max": 5.0,
        "median": 5.0,
        "n": 14
    },
    "Vitamin B2 (%w/v)": {
        "min": 0.02,
        "max": 0.2,
        "median": 0.02,
        "n": 20
    },
    "VEGF (%w/v)": {
        "min": 0.01,
        "max": 10.0,
        "median": 1.0,
        "n": 6
    },
    "Polypyrrole:PSS (%w/v)": {
        "min": 0.1,
        "max": 0.4,
        "median": 0.2,
        "n": 3
    },
    "borate bioactive glass (%w/v)": {
        "min": 0.1,
        "max": 0.1,
        "median": 0.1,
        "n": 1
    },
    "astaxanthin (%w/v)": {
        "min": 0.01,
        "max": 0.01,
        "median": 0.01,
        "n": 1
    },
    "PRP (%v/v)": {
        "min": 20.0,
        "max": 20.0,
        "median": 20.0,
        "n": 3
    },
    "methacrylated collagen (%w/v)": {
        "min": 0.2,
        "max": 50.0,
        "median": 0.48,
        "n": 25
    },
    "\u03b1-Toc (\u00b5M)": {
        "min": 100.0,
        "max": 100.0,
        "median": 100.0,
        "n": 5
    },
    "ascorbic acid (mM)": {
        "min": 3.4,
        "max": 3.4,
        "median": 3.4,
        "n": 5
    },
    "Liver dECM (%w/v)": {
        "min": 0.5,
        "max": 100.0,
        "median": 2.0,
        "n": 139
    },
    "galactosylated alginate (%w/v)": {
        "min": 0.375,
        "max": 1.0,
        "median": 1.0,
        "n": 22
    },
    "SC-PEG (%w/v)": {
        "min": 1.44,
        "max": 1.44,
        "median": 1.44,
        "n": 4
    },
    "SFMA-L (%w/v)": {
        "min": 10.0,
        "max": 10.0,
        "median": 10.0,
        "n": 1
    },
    "SFMA-M (%w/v)": {
        "min": 10.0,
        "max": 10.0,
        "median": 10.0,
        "n": 1
    },
    "SFMA-H (%w/v)": {
        "min": 10.0,
        "max": 10.0,
        "median": 10.0,
        "n": 1
    },
    "KdECMMA (%w/v)": {
        "min": 1.0,
        "max": 3.0,
        "median": 2.0,
        "n": 15
    },
    "BA silk fibroin (%w/v)": {
        "min": 0.5,
        "max": 3.0,
        "median": 1.5,
        "n": 32
    },
    "Carrageenan (%v)": {
        "min": 0.5,
        "max": 1.5,
        "median": 1.0,
        "n": 43
    },
    "Carbopol ETD 2020 NF (%w/v)": {
        "min": 0.1,
        "max": 1.2,
        "median": 0.5,
        "n": 32
    },
    "Carbopol Ultrez 10 NF (%w/v)": {
        "min": 1.5,
        "max": 1.5,
        "median": 1.5,
        "n": 6
    },
    "Carbopol NF-980 (%w/v)": {
        "min": 1.2,
        "max": 1.2,
        "median": 1.2,
        "n": 4
    },
    "FBS (%v/v)": {
        "min": 10.0,
        "max": 10.0,
        "median": 10.0,
        "n": 34
    },
    "MeTro (%w/v)": {
        "min": 7.5,
        "max": 7.5,
        "median": 7.5,
        "n": 28
    },
    "Triethanolamine (%v/v)": {
        "min": 3.0,
        "max": 3.0,
        "median": 3.0,
        "n": 4
    },
    "PEG-Fibrinogen (%w/v)": {
        "min": 1.0,
        "max": 1.0,
        "median": 1.0,
        "n": 8
    },
    "polyethylene glycol dimethacrylate (%w/v)": {
        "min": 1.0,
        "max": 1.0,
        "median": 1.0,
        "n": 50
    },
    "aprotinin (\u00b5g/ml)": {
        "min": 0.2,
        "max": 10.0,
        "median": 0.2,
        "n": 3
    },
    "gold nanorod (mg/mL)": {
        "min": 0.1,
        "max": 0.1,
        "median": 0.1,
        "n": 24
    },
    "egg white (w/v)": {
        "min": 1.0,
        "max": 3.0,
        "median": 2.0,
        "n": 10
    },
    "1-Vinyl-2-pyrrolidone (v/v)": {
        "min": 0.75,
        "max": 0.75,
        "median": 0.75,
        "n": 4
    },
    "carboxyl functionalized carbon nanotubes (%w/v)": {
        "min": 0.3,
        "max": 2.0,
        "median": 1.0,
        "n": 9
    },
    "polyHIPE (%w/v)": {
        "min": 1.0,
        "max": 5.0,
        "median": 3.0,
        "n": 5
    },
    "\u03b2-D galactose (mM)": {
        "min": 40.0,
        "max": 40.0,
        "median": 40.0,
        "n": 2
    },
    "hydrogen peroxide (H2O2) (%v/v)": {
        "min": 0.09,
        "max": 3.87,
        "median": 3.87,
        "n": 55
    },
    "lactic acid (v/v)": {
        "min": 3.0,
        "max": 3.0,
        "median": 3.0,
        "n": 30
    },
    "NorCol (%w/v)": {
        "min": 0.2,
        "max": 1.0,
        "median": 0.6,
        "n": 13
    },
    "DTT (%w/v)": {
        "min": 0.5,
        "max": 8.0,
        "median": 0.7,
        "n": 10
    },
    "ammonium persulfate (mM)": {
        "min": 3.75,
        "max": 5.0,
        "median": 4.375,
        "n": 16
    },
    "diTyr-RGD (mM)": {
        "min": 2.0,
        "max": 6.0,
        "median": 4.0,
        "n": 8
    },
    "PHEG-Tyr (%w/v)": {
        "min": 10.0,
        "max": 10.0,
        "median": 10.0,
        "n": 12
    },
    "MMP2-degradable peptide (%w/v)": {
        "min": 0.1,
        "max": 0.5,
        "median": 0.3,
        "n": 8
    },
    "KdECM (%w/v)": {
        "min": 1.0,
        "max": 5.0,
        "median": 3.3,
        "n": 11
    },
    "EDC (mg)": {
        "min": 50.0,
        "max": 50.0,
        "median": 50.0,
        "n": 22
    },
    "NHS (mg)": {
        "min": 10.0,
        "max": 25.0,
        "median": 10.0,
        "n": 22
    },
    "VA086 (%w/v)": {
        "min": 0.5,
        "max": 20.0,
        "median": 0.75,
        "n": 10
    },
    "PGS (%w/v)": {
        "min": 2.0,
        "max": 20.0,
        "median": 11.0,
        "n": 2
    },
    "thiolated HA (%w/v)": {
        "min": 0.04,
        "max": 0.067,
        "median": 0.05,
        "n": 10
    },
    "boron nitride nanotubes (%w/v)": {
        "min": 0.05,
        "max": 0.1,
        "median": 0.075,
        "n": 3
    },
    "PEDOT:PSS (ul)": {
        "min": 0.1,
        "max": 0.3,
        "median": 0.1,
        "n": 18
    },
    "KCl (mM)": {
        "min": 5.0,
        "max": 5.0,
        "median": 5.0,
        "n": 15
    },
    "skeletal muscle ECM methacrylate (%w/v)": {
        "min": 3.0,
        "max": 3.0,
        "median": 3.0,
        "n": 7
    },
    "PEO (%w/v)": {
        "min": 0.4,
        "max": 0.4,
        "median": 0.4,
        "n": 9
    },
    "Carbon dots (mg/ml)": {
        "min": 20.0,
        "max": 20.0,
        "median": 20.0,
        "n": 9
    },
    "Laminin (ug/ml)": {
        "min": 10.0,
        "max": 93.75,
        "median": 70.0,
        "n": 13
    },
    "DF-PEG (%w/v)": {
        "min": 25.0,
        "max": 25.0,
        "median": 25.0,
        "n": 6
    },
    "omentum ECM (%w/v)": {
        "min": 2.0,
        "max": 2.0,
        "median": 2.0,
        "n": 2
    },
    "thrombin (unit/ml)": {
        "min": 2.5,
        "max": 50.0,
        "median": 10.0,
        "n": 41
    },
    "Carbon nanotube (CNT) (w/v)": {
        "min": 0.5,
        "max": 6.0,
        "median": 1.5,
        "n": 18
    },
    "Phytagel (%v)": {
        "min": 0.3,
        "max": 0.3,
        "median": 0.3,
        "n": 1
    },
    "Laponite-XLG (%w/w)": {
        "min": 2.3,
        "max": 2.3,
        "median": 2.3,
        "n": 25
    },
    "sodium carboxymethyl cellulose (mg)": {
        "min": 75.0,
        "max": 150.0,
        "median": 100.0,
        "n": 60
    },
    "lysozyme amyloid nanofibrils:gold nanoparticles (mg)": {
        "min": 2.5,
        "max": 37.5,
        "median": 12.5,
        "n": 6
    },
    "Lysozyme amyloid nanofibrils (mg)": {
        "min": 2.5,
        "max": 37.5,
        "median": 12.5,
        "n": 31
    },
    "rGO (mg/ml)": {
        "min": 150.0,
        "max": 150.0,
        "median": 150.0,
        "n": 2
    },
    "Methacrylated gellan gum (%w/v)": {
        "min": 2.0,
        "max": 3.5,
        "median": 3.0,
        "n": 15
    },
    "Acetylsalicylic Acid (%w/w)": {
        "min": 10.0,
        "max": 10.0,
        "median": 10.0,
        "n": 3
    },
    "PVA methacrylate (%w/v)": {
        "min": 1.0,
        "max": 5.0,
        "median": 5.0,
        "n": 11
    },
    "MXene (mg/ml)": {
        "min": 0.1,
        "max": 5.0,
        "median": 0.75,
        "n": 6
    }
}

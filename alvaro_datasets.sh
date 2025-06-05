## technote-cart-fmc63-v2.0


cd ~/pixelgen/PixelGen/datasets/technote-cart-fmc63-v2.0

if [ $? -eq 0 ]; then
    # Panel file
    curl -O https://pixelgen-technologies-datasets.s3.eu-north-1.amazonaws.com/mpx-datasets/scsp/2.0/technote-cart-fmc63-v2.0/human-sc-immunology-spatial-proteomics-2_fmc63.csv

    # Samplesheet
    curl -O https://pixelgen-technologies-datasets.s3.eu-north-1.amazonaws.com/mpx-datasets/scsp/2.0/technote-cart-fmc63-v2.0/samplesheet.technote-cart-fmc63-v2.0.csv

    # Input files
    curl -O https://pixelgen-technologies-datasets.s3.eu-north-1.amazonaws.com/mpx-datasets/scsp/2.0/technote-cart-fmc63-v2.0/Sample01_human_pbmcs_control_R1_001.fastq.gz
    curl -O https://pixelgen-technologies-datasets.s3.eu-north-1.amazonaws.com/mpx-datasets/scsp/2.0/technote-cart-fmc63-v2.0/Sample01_human_pbmcs_control_R2_001.fastq.gz
    curl -O https://pixelgen-technologies-datasets.s3.eu-north-1.amazonaws.com/mpx-datasets/scsp/2.0/technote-cart-fmc63-v2.0/Sample02_Raji_control_R1_001.fastq.gz
    curl -O https://pixelgen-technologies-datasets.s3.eu-north-1.amazonaws.com/mpx-datasets/scsp/2.0/technote-cart-fmc63-v2.0/Sample02_Raji_control_R2_001.fastq.gz
    curl -O https://pixelgen-technologies-datasets.s3.eu-north-1.amazonaws.com/mpx-datasets/scsp/2.0/technote-cart-fmc63-v2.0/Sample03_CART_control_R1_001.fastq.gz
    curl -O https://pixelgen-technologies-datasets.s3.eu-north-1.amazonaws.com/mpx-datasets/scsp/2.0/technote-cart-fmc63-v2.0/Sample03_CART_control_R2_001.fastq.gz
    curl -O https://pixelgen-technologies-datasets.s3.eu-north-1.amazonaws.com/mpx-datasets/scsp/2.0/technote-cart-fmc63-v2.0/Sample04_CART_Raji_co-culture_4h_R1_001.fastq.gz
    curl -O https://pixelgen-technologies-datasets.s3.eu-north-1.amazonaws.com/mpx-datasets/scsp/2.0/technote-cart-fmc63-v2.0/Sample04_CART_Raji_co-culture_4h_R2_001.fastq.gz
    curl -O https://pixelgen-technologies-datasets.s3.eu-north-1.amazonaws.com/mpx-datasets/scsp/2.0/technote-cart-fmc63-v2.0/Sample05_CART_Raji_co-culture_24h_R1_001.fastq.gz
    curl -O https://pixelgen-technologies-datasets.s3.eu-north-1.amazonaws.com/mpx-datasets/scsp/2.0/technote-cart-fmc63-v2.0/Sample05_CART_Raji_co-culture_24h_R2_001.fastq.gz

else
    echo technote-cart-fmc63-v2.0 FAIL
fi

## 4-donors-pbmcs-v2.0

cd ~/pixelgen/PixelGen/datasets/4-donors-pbmcs-v2.0

if [ $? -eq 0 ]; then

    # Panel file
    curl -O https://pixelgen-technologies-datasets.s3.eu-north-1.amazonaws.com/mpx-datasets/scsp/2.0/4-donors-pbmcs-v2.0/human-sc-immunology-spatial-proteomics-2-v0.1.0.csv

    # Samplesheets
    curl -O https://pixelgen-technologies-datasets.s3.eu-north-1.amazonaws.com/mpx-datasets/scsp/2.0/4-donors-pbmcs-v2.0/P4/samplesheet.4-donors-pbmcs-v2.0-P4.csv
    curl -O https://pixelgen-technologies-datasets.s3.eu-north-1.amazonaws.com/mpx-datasets/scsp/2.0/4-donors-pbmcs-v2.0/10B/samplesheet.4-donors-pbmcs-v2.0-10B.csv
    curl -O https://pixelgen-technologies-datasets.s3.eu-north-1.amazonaws.com/mpx-datasets/scsp/2.0/4-donors-pbmcs-v2.0/S4/samplesheet.4-donors-pbmcs-v2.0-S4.csv

    # Input files

    ##P4
    curl -O https://pixelgen-technologies-datasets.s3.eu-north-1.amazonaws.com/mpx-datasets/scsp/2.0/4-donors-pbmcs-v2.0/P4/Sample01_PBMC_1085_r1_R1_001.fastq.gz
    curl -O https://pixelgen-technologies-datasets.s3.eu-north-1.amazonaws.com/mpx-datasets/scsp/2.0/4-donors-pbmcs-v2.0/P4/Sample01_PBMC_1085_r1_R2_001.fastq.gz
    curl -O https://pixelgen-technologies-datasets.s3.eu-north-1.amazonaws.com/mpx-datasets/scsp/2.0/4-donors-pbmcs-v2.0/P4/Sample02_PBMC_8967_r1_R1_001.fastq.gz
    curl -O https://pixelgen-technologies-datasets.s3.eu-north-1.amazonaws.com/mpx-datasets/scsp/2.0/4-donors-pbmcs-v2.0/P4/Sample02_PBMC_8967_r1_R2_001.fastq.gz
    curl -O https://pixelgen-technologies-datasets.s3.eu-north-1.amazonaws.com/mpx-datasets/scsp/2.0/4-donors-pbmcs-v2.0/P4/Sample03_PBMC_8547_r1_R1_001.fastq.gz
    curl -O https://pixelgen-technologies-datasets.s3.eu-north-1.amazonaws.com/mpx-datasets/scsp/2.0/4-donors-pbmcs-v2.0/P4/Sample03_PBMC_8547_r1_R2_001.fastq.gz
    curl -O https://pixelgen-technologies-datasets.s3.eu-north-1.amazonaws.com/mpx-datasets/scsp/2.0/4-donors-pbmcs-v2.0/P4/Sample05_PBMC_8549_r1_R1_001.fastq.gz
    curl -O https://pixelgen-technologies-datasets.s3.eu-north-1.amazonaws.com/mpx-datasets/scsp/2.0/4-donors-pbmcs-v2.0/P4/Sample05_PBMC_8549_r1_R2_001.fastq.gz
    curl -O https://pixelgen-technologies-datasets.s3.eu-north-1.amazonaws.com/mpx-datasets/scsp/2.0/4-donors-pbmcs-v2.0/P4/Sample06_PBMC_8549_r2_R1_001.fastq.gz
    curl -O https://pixelgen-technologies-datasets.s3.eu-north-1.amazonaws.com/mpx-datasets/scsp/2.0/4-donors-pbmcs-v2.0/P4/Sample06_PBMC_8549_r2_R2_001.fastq.gz
    curl -O https://pixelgen-technologies-datasets.s3.eu-north-1.amazonaws.com/mpx-datasets/scsp/2.0/4-donors-pbmcs-v2.0/P4/Sample07_PHA_PBMC_8549_r1_R1_001.fastq.gz
    curl -O https://pixelgen-technologies-datasets.s3.eu-north-1.amazonaws.com/mpx-datasets/scsp/2.0/4-donors-pbmcs-v2.0/P4/Sample07_PHA_PBMC_8549_r1_R2_001.fastq.gz
    curl -O https://pixelgen-technologies-datasets.s3.eu-north-1.amazonaws.com/mpx-datasets/scsp/2.0/4-donors-pbmcs-v2.0/P4/Sample08_PHA_PBMC_8549_r2_R1_001.fastq.gz
    curl -O https://pixelgen-technologies-datasets.s3.eu-north-1.amazonaws.com/mpx-datasets/scsp/2.0/4-donors-pbmcs-v2.0/P4/Sample08_PHA_PBMC_8549_r2_R2_001.fastq.gz
    curl -O https://pixelgen-technologies-datasets.s3.eu-north-1.amazonaws.com/mpx-datasets/scsp/2.0/4-donors-pbmcs-v2.0/P4/Sample09_PBMC_1085_r2_R1_001.fastq.gz
    curl -O https://pixelgen-technologies-datasets.s3.eu-north-1.amazonaws.com/mpx-datasets/scsp/2.0/4-donors-pbmcs-v2.0/P4/Sample09_PBMC_1085_r2_R2_001.fastq.gz
    curl -O https://pixelgen-technologies-datasets.s3.eu-north-1.amazonaws.com/mpx-datasets/scsp/2.0/4-donors-pbmcs-v2.0/P4/Sample10_PBMC_8967_r2_R1_001.fastq.gz
    curl -O https://pixelgen-technologies-datasets.s3.eu-north-1.amazonaws.com/mpx-datasets/scsp/2.0/4-donors-pbmcs-v2.0/P4/Sample10_PBMC_8967_r2_R2_001.fastq.gz
    curl -O https://pixelgen-technologies-datasets.s3.eu-north-1.amazonaws.com/mpx-datasets/scsp/2.0/4-donors-pbmcs-v2.0/P4/Sample11_PBMC_8547_r2_R1_001.fastq.gz
    curl -O https://pixelgen-technologies-datasets.s3.eu-north-1.amazonaws.com/mpx-datasets/scsp/2.0/4-donors-pbmcs-v2.0/P4/Sample11_PBMC_8547_r2_R2_001.fastq.gz

    ##10B
    curl -O https://pixelgen-technologies-datasets.s3.eu-north-1.amazonaws.com/mpx-datasets/scsp/2.0/4-donors-pbmcs-v2.0/10B/Sample01_PBMC_1085_r1_R1_001.fastq.gz
    curl -O https://pixelgen-technologies-datasets.s3.eu-north-1.amazonaws.com/mpx-datasets/scsp/2.0/4-donors-pbmcs-v2.0/10B/Sample01_PBMC_1085_r1_R2_001.fastq.gz
    curl -O https://pixelgen-technologies-datasets.s3.eu-north-1.amazonaws.com/mpx-datasets/scsp/2.0/4-donors-pbmcs-v2.0/10B/Sample02_PBMC_8967_r1_R1_001.fastq.gz
    curl -O https://pixelgen-technologies-datasets.s3.eu-north-1.amazonaws.com/mpx-datasets/scsp/2.0/4-donors-pbmcs-v2.0/10B/Sample02_PBMC_8967_r1_R2_001.fastq.gz
    curl -O https://pixelgen-technologies-datasets.s3.eu-north-1.amazonaws.com/mpx-datasets/scsp/2.0/4-donors-pbmcs-v2.0/10B/Sample03_PBMC_8547_r1_R1_001.fastq.gz
    curl -O https://pixelgen-technologies-datasets.s3.eu-north-1.amazonaws.com/mpx-datasets/scsp/2.0/4-donors-pbmcs-v2.0/10B/Sample03_PBMC_8547_r1_R2_001.fastq.gz
    curl -O https://pixelgen-technologies-datasets.s3.eu-north-1.amazonaws.com/mpx-datasets/scsp/2.0/4-donors-pbmcs-v2.0/10B/Sample09_PBMC_1085_r2_R1_001.fastq.gz
    curl -O https://pixelgen-technologies-datasets.s3.eu-north-1.amazonaws.com/mpx-datasets/scsp/2.0/4-donors-pbmcs-v2.0/10B/Sample09_PBMC_1085_r2_R2_001.fastq.gz
    curl -O https://pixelgen-technologies-datasets.s3.eu-north-1.amazonaws.com/mpx-datasets/scsp/2.0/4-donors-pbmcs-v2.0/10B/Sample10_PBMC_8967_r2_R1_001.fastq.gz
    curl -O https://pixelgen-technologies-datasets.s3.eu-north-1.amazonaws.com/mpx-datasets/scsp/2.0/4-donors-pbmcs-v2.0/10B/Sample10_PBMC_8967_r2_R2_001.fastq.gz
    curl -O https://pixelgen-technologies-datasets.s3.eu-north-1.amazonaws.com/mpx-datasets/scsp/2.0/4-donors-pbmcs-v2.0/10B/Sample11_PBMC_8547_r2_R1_001.fastq.gz
    curl -O https://pixelgen-technologies-datasets.s3.eu-north-1.amazonaws.com/mpx-datasets/scsp/2.0/4-donors-pbmcs-v2.0/10B/Sample11_PBMC_8547_r2_R2_001.fastq.gz

    ##S4
    curl -O https://pixelgen-technologies-datasets.s3.eu-north-1.amazonaws.com/mpx-datasets/scsp/2.0/4-donors-pbmcs-v2.0/Sample01_PBMC_1085_r1_R1_001.fastq.gz
    curl -O https://pixelgen-technologies-datasets.s3.eu-north-1.amazonaws.com/mpx-datasets/scsp/2.0/4-donors-pbmcs-v2.0/Sample01_PBMC_1085_r1_R2_001.fastq.gz
    curl -O https://pixelgen-technologies-datasets.s3.eu-north-1.amazonaws.com/mpx-datasets/scsp/2.0/4-donors-pbmcs-v2.0/Sample02_PBMC_8967_r1_R1_001.fastq.gz
    curl -O https://pixelgen-technologies-datasets.s3.eu-north-1.amazonaws.com/mpx-datasets/scsp/2.0/4-donors-pbmcs-v2.0/Sample02_PBMC_8967_r1_R2_001.fastq.gz
    curl -O https://pixelgen-technologies-datasets.s3.eu-north-1.amazonaws.com/mpx-datasets/scsp/2.0/4-donors-pbmcs-v2.0/Sample03_PBMC_8547_r1_R1_001.fastq.gz
    curl -O https://pixelgen-technologies-datasets.s3.eu-north-1.amazonaws.com/mpx-datasets/scsp/2.0/4-donors-pbmcs-v2.0/Sample03_PBMC_8547_r1_R2_001.fastq.gz
    curl -O https://pixelgen-technologies-datasets.s3.eu-north-1.amazonaws.com/mpx-datasets/scsp/2.0/4-donors-pbmcs-v2.0/Sample09_PBMC_1085_r2_R1_001.fastq.gz
    curl -O https://pixelgen-technologies-datasets.s3.eu-north-1.amazonaws.com/mpx-datasets/scsp/2.0/4-donors-pbmcs-v2.0/Sample09_PBMC_1085_r2_R2_001.fastq.gz
    curl -O https://pixelgen-technologies-datasets.s3.eu-north-1.amazonaws.com/mpx-datasets/scsp/2.0/4-donors-pbmcs-v2.0/Sample10_PBMC_8967_r2_R1_001.fastq.gz
    curl -O https://pixelgen-technologies-datasets.s3.eu-north-1.amazonaws.com/mpx-datasets/scsp/2.0/4-donors-pbmcs-v2.0/Sample10_PBMC_8967_r2_R2_001.fastq.gz
    curl -O https://pixelgen-technologies-datasets.s3.eu-north-1.amazonaws.com/mpx-datasets/scsp/2.0/4-donors-pbmcs-v2.0/Sample11_PBMC_8547_r2_R1_001.fastq.gz
    curl -O https://pixelgen-technologies-datasets.s3.eu-north-1.amazonaws.com/mpx-datasets/scsp/2.0/4-donors-pbmcs-v2.0/Sample11_PBMC_8547_r2_R2_001.fastq.gz

else
    echo 4-donors-pbmcs-v2.0 FAIL
fi


## 5-cell-lines-v2.0

cd ~/pixelgen/PixelGen/datasets/5-cell-lines-v2.0

if [ $? -eq 0 ]; then
    # Panel file
    curl -O https://pixelgen-technologies-datasets.s3.eu-north-1.amazonaws.com/mpx-datasets/scsp/2.0/5-cell-lines-v2.0/human-sc-immunology-spatial-proteomics-2-v0.1.0.csv

    # Samplesheets
    curl -O https://pixelgen-technologies-datasets.s3.eu-north-1.amazonaws.com/mpx-datasets/scsp/2.0/5-cell-lines-v2.0/P4/samplesheet.5-cell-lines-v2.0-P4.csv
    curl -O https://pixelgen-technologies-datasets.s3.eu-north-1.amazonaws.com/mpx-datasets/scsp/2.0/5-cell-lines-v2.0/10B/samplesheet.5-cell-lines-v2.0-10B.csv
    curl -O https://pixelgen-technologies-datasets.s3.eu-north-1.amazonaws.com/mpx-datasets/scsp/2.0/5-cell-lines-v2.0/S4/samplesheet.5-cell-lines-v2.0-S4.csv

    # Input files

    ##P4
    curl -O https://pixelgen-technologies-datasets.s3.eu-north-1.amazonaws.com/mpx-datasets/scsp/2.0/5-cell-lines-v2.0/P4/Sample04_Daudi_r1_R1_001.fastq.gz
    curl -O https://pixelgen-technologies-datasets.s3.eu-north-1.amazonaws.com/mpx-datasets/scsp/2.0/5-cell-lines-v2.0/P4/Sample04_Daudi_r1_R2_001.fastq.gz
    curl -O https://pixelgen-technologies-datasets.s3.eu-north-1.amazonaws.com/mpx-datasets/scsp/2.0/5-cell-lines-v2.0/P4/Sample05_Raji_r1_R1_001.fastq.gz
    curl -O https://pixelgen-technologies-datasets.s3.eu-north-1.amazonaws.com/mpx-datasets/scsp/2.0/5-cell-lines-v2.0/P4/Sample05_Raji_r1_R2_001.fastq.gz
    curl -O https://pixelgen-technologies-datasets.s3.eu-north-1.amazonaws.com/mpx-datasets/scsp/2.0/5-cell-lines-v2.0/P4/Sample06_Ramos_r1_R1_001.fastq.gz
    curl -O https://pixelgen-technologies-datasets.s3.eu-north-1.amazonaws.com/mpx-datasets/scsp/2.0/5-cell-lines-v2.0/P4/Sample06_Ramos_r1_R2_001.fastq.gz
    curl -O https://pixelgen-technologies-datasets.s3.eu-north-1.amazonaws.com/mpx-datasets/scsp/2.0/5-cell-lines-v2.0/P4/Sample07_SupT1_r1_R1_001.fastq.gz
    curl -O https://pixelgen-technologies-datasets.s3.eu-north-1.amazonaws.com/mpx-datasets/scsp/2.0/5-cell-lines-v2.0/P4/Sample07_SupT1_r1_R2_001.fastq.gz
    curl -O https://pixelgen-technologies-datasets.s3.eu-north-1.amazonaws.com/mpx-datasets/scsp/2.0/5-cell-lines-v2.0/P4/Sample08_Thp1_r1_R1_001.fastq.gz
    curl -O https://pixelgen-technologies-datasets.s3.eu-north-1.amazonaws.com/mpx-datasets/scsp/2.0/5-cell-lines-v2.0/P4/Sample08_Thp1_r1_R2_001.fastq.gz
    curl -O https://pixelgen-technologies-datasets.s3.eu-north-1.amazonaws.com/mpx-datasets/scsp/2.0/5-cell-lines-v2.0/P4/Sample12_Daudi_r2_R1_001.fastq.gz
    curl -O https://pixelgen-technologies-datasets.s3.eu-north-1.amazonaws.com/mpx-datasets/scsp/2.0/5-cell-lines-v2.0/P4/Sample12_Daudi_r2_R2_001.fastq.gz
    curl -O https://pixelgen-technologies-datasets.s3.eu-north-1.amazonaws.com/mpx-datasets/scsp/2.0/5-cell-lines-v2.0/P4/Sample13_Raji_r2_R1_001.fastq.gz
    curl -O https://pixelgen-technologies-datasets.s3.eu-north-1.amazonaws.com/mpx-datasets/scsp/2.0/5-cell-lines-v2.0/P4/Sample13_Raji_r2_R2_001.fastq.gz
    curl -O https://pixelgen-technologies-datasets.s3.eu-north-1.amazonaws.com/mpx-datasets/scsp/2.0/5-cell-lines-v2.0/P4/Sample14_Ramos_r2_R1_001.fastq.gz
    curl -O https://pixelgen-technologies-datasets.s3.eu-north-1.amazonaws.com/mpx-datasets/scsp/2.0/5-cell-lines-v2.0/P4/Sample14_Ramos_r2_R2_001.fastq.gz
    curl -O https://pixelgen-technologies-datasets.s3.eu-north-1.amazonaws.com/mpx-datasets/scsp/2.0/5-cell-lines-v2.0/P4/Sample15_SupT1_r2_R1_001.fastq.gz
    curl -O https://pixelgen-technologies-datasets.s3.eu-north-1.amazonaws.com/mpx-datasets/scsp/2.0/5-cell-lines-v2.0/P4/Sample15_SupT1_r2_R2_001.fastq.gz
    curl -O https://pixelgen-technologies-datasets.s3.eu-north-1.amazonaws.com/mpx-datasets/scsp/2.0/5-cell-lines-v2.0/P4/Sample16_Thp1_r2_R1_001.fastq.gz
    curl -O https://pixelgen-technologies-datasets.s3.eu-north-1.amazonaws.com/mpx-datasets/scsp/2.0/5-cell-lines-v2.0/P4/Sample16_Thp1_r2_R2_001.fastq.gz

    ##10B
    curl -O https://pixelgen-technologies-datasets.s3.eu-north-1.amazonaws.com/mpx-datasets/scsp/2.0/5-cell-lines-v2.0/10B/Sample04_Daudi_r1_R1_001.fastq.gz
    curl -O https://pixelgen-technologies-datasets.s3.eu-north-1.amazonaws.com/mpx-datasets/scsp/2.0/5-cell-lines-v2.0/10B/Sample04_Daudi_r1_R2_001.fastq.gz
    curl -O https://pixelgen-technologies-datasets.s3.eu-north-1.amazonaws.com/mpx-datasets/scsp/2.0/5-cell-lines-v2.0/10B/Sample05_Raji_r1_R1_001.fastq.gz
    curl -O https://pixelgen-technologies-datasets.s3.eu-north-1.amazonaws.com/mpx-datasets/scsp/2.0/5-cell-lines-v2.0/10B/Sample05_Raji_r1_R2_001.fastq.gz
    curl -O https://pixelgen-technologies-datasets.s3.eu-north-1.amazonaws.com/mpx-datasets/scsp/2.0/5-cell-lines-v2.0/10B/Sample06_Ramos_r1_R1_001.fastq.gz
    curl -O https://pixelgen-technologies-datasets.s3.eu-north-1.amazonaws.com/mpx-datasets/scsp/2.0/5-cell-lines-v2.0/10B/Sample06_Ramos_r1_R2_001.fastq.gz
    curl -O https://pixelgen-technologies-datasets.s3.eu-north-1.amazonaws.com/mpx-datasets/scsp/2.0/5-cell-lines-v2.0/10B/Sample07_SupT1_r1_R1_001.fastq.gz
    curl -O https://pixelgen-technologies-datasets.s3.eu-north-1.amazonaws.com/mpx-datasets/scsp/2.0/5-cell-lines-v2.0/10B/Sample07_SupT1_r1_R2_001.fastq.gz
    curl -O https://pixelgen-technologies-datasets.s3.eu-north-1.amazonaws.com/mpx-datasets/scsp/2.0/5-cell-lines-v2.0/10B/Sample08_Thp1_r1_R1_001.fastq.gz
    curl -O https://pixelgen-technologies-datasets.s3.eu-north-1.amazonaws.com/mpx-datasets/scsp/2.0/5-cell-lines-v2.0/10B/Sample08_Thp1_r1_R2_001.fastq.gz
    curl -O https://pixelgen-technologies-datasets.s3.eu-north-1.amazonaws.com/mpx-datasets/scsp/2.0/5-cell-lines-v2.0/10B/Sample12_Daudi_r2_R1_001.fastq.gz
    curl -O https://pixelgen-technologies-datasets.s3.eu-north-1.amazonaws.com/mpx-datasets/scsp/2.0/5-cell-lines-v2.0/10B/Sample12_Daudi_r2_R2_001.fastq.gz
    curl -O https://pixelgen-technologies-datasets.s3.eu-north-1.amazonaws.com/mpx-datasets/scsp/2.0/5-cell-lines-v2.0/10B/Sample13_Raji_r2_R1_001.fastq.gz
    curl -O https://pixelgen-technologies-datasets.s3.eu-north-1.amazonaws.com/mpx-datasets/scsp/2.0/5-cell-lines-v2.0/10B/Sample13_Raji_r2_R2_001.fastq.gz
    curl -O https://pixelgen-technologies-datasets.s3.eu-north-1.amazonaws.com/mpx-datasets/scsp/2.0/5-cell-lines-v2.0/10B/Sample14_Ramos_r2_R1_001.fastq.gz
    curl -O https://pixelgen-technologies-datasets.s3.eu-north-1.amazonaws.com/mpx-datasets/scsp/2.0/5-cell-lines-v2.0/10B/Sample14_Ramos_r2_R2_001.fastq.gz
    curl -O https://pixelgen-technologies-datasets.s3.eu-north-1.amazonaws.com/mpx-datasets/scsp/2.0/5-cell-lines-v2.0/10B/Sample15_SupT1_r2_R1_001.fastq.gz
    curl -O https://pixelgen-technologies-datasets.s3.eu-north-1.amazonaws.com/mpx-datasets/scsp/2.0/5-cell-lines-v2.0/10B/Sample15_SupT1_r2_R2_001.fastq.gz
    curl -O https://pixelgen-technologies-datasets.s3.eu-north-1.amazonaws.com/mpx-datasets/scsp/2.0/5-cell-lines-v2.0/10B/Sample16_Thp1_r2_R1_001.fastq.gz
    curl -O https://pixelgen-technologies-datasets.s3.eu-north-1.amazonaws.com/mpx-datasets/scsp/2.0/5-cell-lines-v2.0/10B/Sample16_Thp1_r2_R2_001.fastq.gz

    ##S4
    curl -O https://pixelgen-technologies-datasets.s3.eu-north-1.amazonaws.com/mpx-datasets/scsp/2.0/5-cell-lines-v2.0/S4/Sample04_Daudi_r1_R1_001.fastq.gz
    curl -O https://pixelgen-technologies-datasets.s3.eu-north-1.amazonaws.com/mpx-datasets/scsp/2.0/5-cell-lines-v2.0/S4/Sample04_Daudi_r1_R2_001.fastq.gz
    curl -O https://pixelgen-technologies-datasets.s3.eu-north-1.amazonaws.com/mpx-datasets/scsp/2.0/5-cell-lines-v2.0/S4/Sample05_Raji_r1_R1_001.fastq.gz
    curl -O https://pixelgen-technologies-datasets.s3.eu-north-1.amazonaws.com/mpx-datasets/scsp/2.0/5-cell-lines-v2.0/S4/Sample05_Raji_r1_R2_001.fastq.gz
    curl -O https://pixelgen-technologies-datasets.s3.eu-north-1.amazonaws.com/mpx-datasets/scsp/2.0/5-cell-lines-v2.0/S4/Sample06_Ramos_r1_R1_001.fastq.gz
    curl -O https://pixelgen-technologies-datasets.s3.eu-north-1.amazonaws.com/mpx-datasets/scsp/2.0/5-cell-lines-v2.0/S4/Sample06_Ramos_r1_R2_001.fastq.gz
    curl -O https://pixelgen-technologies-datasets.s3.eu-north-1.amazonaws.com/mpx-datasets/scsp/2.0/5-cell-lines-v2.0/S4/Sample07_SupT1_r1_R1_001.fastq.gz
    curl -O https://pixelgen-technologies-datasets.s3.eu-north-1.amazonaws.com/mpx-datasets/scsp/2.0/5-cell-lines-v2.0/S4/Sample07_SupT1_r1_R2_001.fastq.gz
    curl -O https://pixelgen-technologies-datasets.s3.eu-north-1.amazonaws.com/mpx-datasets/scsp/2.0/5-cell-lines-v2.0/S4/Sample08_Thp1_r1_R1_001.fastq.gz
    curl -O https://pixelgen-technologies-datasets.s3.eu-north-1.amazonaws.com/mpx-datasets/scsp/2.0/5-cell-lines-v2.0/S4/Sample08_Thp1_r1_R2_001.fastq.gz
    curl -O https://pixelgen-technologies-datasets.s3.eu-north-1.amazonaws.com/mpx-datasets/scsp/2.0/5-cell-lines-v2.0/S4/Sample12_Daudi_r2_R1_001.fastq.gz
    curl -O https://pixelgen-technologies-datasets.s3.eu-north-1.amazonaws.com/mpx-datasets/scsp/2.0/5-cell-lines-v2.0/S4/Sample12_Daudi_r2_R2_001.fastq.gz
    curl -O https://pixelgen-technologies-datasets.s3.eu-north-1.amazonaws.com/mpx-datasets/scsp/2.0/5-cell-lines-v2.0/S4/Sample13_Raji_r2_R1_001.fastq.gz
    curl -O https://pixelgen-technologies-datasets.s3.eu-north-1.amazonaws.com/mpx-datasets/scsp/2.0/5-cell-lines-v2.0/S4/Sample13_Raji_r2_R2_001.fastq.gz
    curl -O https://pixelgen-technologies-datasets.s3.eu-north-1.amazonaws.com/mpx-datasets/scsp/2.0/5-cell-lines-v2.0/S4/Sample14_Ramos_r2_R1_001.fastq.gz
    curl -O https://pixelgen-technologies-datasets.s3.eu-north-1.amazonaws.com/mpx-datasets/scsp/2.0/5-cell-lines-v2.0/S4/Sample14_Ramos_r2_R2_001.fastq.gz
    curl -O https://pixelgen-technologies-datasets.s3.eu-north-1.amazonaws.com/mpx-datasets/scsp/2.0/5-cell-lines-v2.0/S4/Sample15_SupT1_r2_R1_001.fastq.gz
    curl -O https://pixelgen-technologies-datasets.s3.eu-north-1.amazonaws.com/mpx-datasets/scsp/2.0/5-cell-lines-v2.0/S4/Sample15_SupT1_r2_R2_001.fastq.gz
    curl -O https://pixelgen-technologies-datasets.s3.eu-north-1.amazonaws.com/mpx-datasets/scsp/2.0/5-cell-lines-v2.0/S4/Sample16_Thp1_r2_R1_001.fastq.gz
    curl -O https://pixelgen-technologies-datasets.s3.eu-north-1.amazonaws.com/mpx-datasets/scsp/2.0/5-cell-lines-v2.0/S4/Sample16_Thp1_r2_R2_001.fastq.gz

else
    echo 5-cell-lines-v2.0 FAIL
fi


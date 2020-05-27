# CNA_origin



We proposed a two-step computational framework called CNA_origin to predict the tissue-of-origin of a tumor from its gene CNA levels. CNA origin set up an intellectual deep-learning network mainly composed of autoencoder and convolution neural network (CNN).


If you want to use CNA_origin, you must have gene-level CNA file and label file.

The use of CNA_origin:
CNA_origin.py  -T PATH_GENE_CNV:        &emsp;File of the gene CNV   <br/>
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;             -G PATH_LABEL:  &emsp;File of the sample label  <br/>
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;                 [-d DIM_NUMBER]:The Number of Features after Dimension Reduction, default:100     <br/>
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;                 [-k K_CROSS_VALIDATION]:&emsp;k fold cross validation, default:10      <br/>
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;                 [-s TRAINING_PART_SCALE]:&emsp;Split scale for train/test,default:0.1      <br/>
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;                [-o OUTPUT_FILE]: &emsp;The result output path         <br/>



These datasets were from primary solid tumor samples released by MSKCC in 2013, which could be downloaded from http://cbio.mskcc.org/cancergenomics/pancan_tcga/   <br/>
for example: python CNA_origin.py -T merge-sample -G merge-group  <br/>

CNA origin was implemented in python 3.7.3 using keras (2.24) with the backend of tensorflow (1.14.0)   <br/>

The program now has a bug that can only be run using CPU (not GPU). We are trying to fix it.

If you have any question,please send email to aliang1229@126.com.&emsp; We will continue to improve the code of CNA_origin.



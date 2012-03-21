addpath('tools');

% 0. Initialization
args = argv();

if nargin<1,
    disp('Format: k-Value [fold=1] [image scaling=1] [PCA d]');
    exit(0);
endif

kValue = str2num(args{1});
maxLabel = 0;

fold=1;
if nargin >1,
    fold = str2num(args{2});
endif

rs = 1;
if nargin>=3
    rs = str2num(args{3});
endif

printf('\n\n*************************\nExperiment: kValue=%d, fold=%i, shrink=%i\n', kValue, fold, rs);
 
% 1. Construct LdaImageDB: Read Images
disp('1. Constructing LDA Image DB...');
LdaImageDb = constructImageDB('LdaImageDb.data', fold, rs, './dataset');

% 2. Dimensionality Reduction: LDA
disp('2. Dimensinality Reduction...')
W_optimal = LDA(fold, LdaImageDb);

% 3. Construct FaceDB: Project Data to low-dimensional space
disp('3. Constructing Face DB after Dimensinality Reduction...')
faceDb = constructFaceDB(LdaImageDb, W_optimal);

% 4. Classification: KNN Nearest Neighbor
disp('4. Classifying using KNN');
[correct, totalTesting] = KNN(W_optimal, faceDb, fold, rs, kValue, './dataset');

disp(sprintf('accuracy = %i / %i = %f%%', correct, totalTesting, correct/totalTesting*100) );

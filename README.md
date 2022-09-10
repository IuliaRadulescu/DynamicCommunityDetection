# SemADyn: Semantics Aware Dynamic Community Detection for Temporally Fluctuating Social Networks
## Build quality dynamic communities from structurally dissimilar network snapshots

## How to run

### To do first: 
1. the database with the comment collections (see the zip file within the DATASET folder) must first be imported into local mongo server in order to run the code;
2. then, the comments and submissions must be organized in twelve hour intervals: python organizeUserInteractionsInIntervals.py
3. clustering must also be performed: python TEXT\ CLUSTERING/clusteringDoc2Vec.py (the doc2vec model is already trained)
4. also the structural clusters must be computed: python buildCommunityGraphClean.py; we offer support for both simple Louvain and inertia Louvain

### Now you're ready to go:

Template:

python buildDynamicCommunitiesHybrid.py -db database_name -sim similarity_threshold -o output_file_name

Example:

python buildDynamicCommunitiesHybrid.py -db communityDetectionWimbledon -sim 0.75 -o RAW_OUTPUTS/hybridDynamicSim75.json
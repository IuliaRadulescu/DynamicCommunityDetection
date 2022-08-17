# SemADyn: Semantics Aware Dynamic Community Detection for Temporally Fluctuating Social Networks
## Build quality dynamic communities from structurally dissimilar network snapshots

### How to run

Note: the database with the comment collections (see the zip file within the DATASET folder) must first be imported into local mongo server in order to run the code

Template:

python buildDynamicCommunitiesHybrid.py -db database_name -sim similarity_threshold -o output_file_name

Example:

python buildDynamicCommunitiesHybrid.py -db communityDetectionWimbledon -sim 0.75 -o RAW_OUTPUTS/hybridDynamicSim75.json
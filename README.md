# Circle-Packing

Circle packing in the unit square using ADMM:

minimize \sum_{i,j} f_{ij}(z_i, z_j) + \sum_i g_i(z_i)

f_{ij}(z_i, z_j) = 0, if ||z_i - z_j|| >= 2R;
                 = infinity, if ||z_i - z_j|| < 2R
                 
g_i(z_i) = 0, if R <= z_i <= 1 - R;
         = infinity, otherwise

#!/usr/bin/env python2
import csv
import pandas as pd
import numpy as np
import pickle
import random

result = {}

df =  pd.read_csv('../data/dataset_for_cell_lstm_22Dec.csv')

grouped = df.groupby(['file_name'])
grouped_df = grouped.groups

def parse_featurevector(cell):
    # until now, we use only binary features
    feature_keys = [
        'all_lower',
        'special_chars',
        'is_numeric',
        'is_string',
        #'alpha', <- dont know what this is
        #'alphanum',
        'is_date',
        #'is_formula',
        'is_aggr_frml',
        'is_other_frml',
        'is_hyperlink',
#length	numeric
#words	numeric
#leading_spaces	numeric
        #'first_char_num',
        #'first_char_special',
        #'capitalized',
        'all_upper',
        'all_lower',
        'special_chars',
        'punctuations',
        'contains_colon',
        'last_is_colon',
        'words_like_total',
        'words_like_table',
        'in_year_range',
#ref_val_num	Maybe
#ref_val_str	Maybe
#ref_val_date	Maybe
        'is_referenced',
#ref_is_aggr_frml	Maybe
#ref_is_other_frml	Maybe
#height	numeric
#width	numeric
        'h_alignment_general',
        'h_alignment_left',
        'h_alignment_center',
        'h_alignment_right',
        'h_alignment_center_select',
        'h_alignment_distributed',
        'v_alignment_top',
        'v_alignment_center',
        'v_alignment_bottom',
        'is_rotated',
#indentation	numeric
        'is_fill_pattern_default',
        'is_wraptext',
        'is_shrinked',
        'is_merged',
        'is_fill_color_default',
        'border_top_none',
        'border_top_thin',
        'border_top_medium',
        'border_top_dashed',
        'border_top_dotted',
        'border_top_thick',
        'border_top_double',
        'border_top_hair',
        'border_bottom_none',
        'border_bottom_thin',
        'border_bottom_medium',
        'border_bottom_dashed',
        'border_bottom_dotted',
        'border_bottom_thick',
        'border_bottom_double',
        'border_bottom_hair',
        'border_left_none',
        'border_left_thin',
        'border_left_medium',
        'border_left_dashed',
        'border_left_dotted',
        'border_left_thick',
        'border_left_double',
        'border_left_hair',
        'border_right_none',
        'border_right_thin',
        'border_right_medium',
        'border_right_dashed',
        'border_right_dotted',
        'border_right_thick',
        'border_right_double',
        'border_right_hair',
        'border_right_med_dashdot',
        'border_right_med_dash2dots',
#num_of_borders	numeric
#font_height	numeric
        'is_font_color_default',
        'is_bold',
        'is_italic',
        'is_underlined',
#dist_from_top	numeric
#dist_from_bottom	numeric
#dist_from_left	numeric
#dist_from_right	numeric
#dist_gap_top	numeric
#dist_gap_bottom	numeric
#dist_gap_left	numeric
#dist_gap_right	numeric
#dist_neighbor_top	numeric
#dist_neighbor_bottom	numeric
#dist_neighbor_left	numeric
#dist_neighbor_right	numeric
#entropy_row_bold	numeric
#entropy_row_fontH	numeric
#entropy_row_halign	numeric
#entropy_row_type	numeric
#entropy_col_spaces	numeric
#entropy_col_indent	numeric
#entropy_col_fontH	numeric
#entropy_col_halign	numeric
#row_sparsity	numeric
#col_sparsity	numeric
#row_coverage	numeric
#col_coverage	numeric
#cell_row_coverage	numeric
#cell_col_coverage	numeric
#row_percent_num	numeric
#row_percent_str	numeric
#row_percent_aggr	numeric
#row_percent_date	numeric
#row_percent_link	numeric
#row_percent_other	numeric
#col_percent_num	numeric
#col_percent_str	numeric
#col_percent_aggr	numeric
#col_percent_date	numeric
#col_percent_link	numeric
#col_percent_other	numeric
        #'is_strong_str_aggr_row',
        #'matches_top_type',
        #'has_top_neighbor',
#count_neighbors_top	numeric
        #'top_neighbor_num',
        #'top_neighbor_str',
        #'top_neighbor_frml',
        #'top_neighbor_date',
        #'top_neighbor_aggr',
        #'top_neighbor_mix',
        #'matches_top_style',
        #'matches_bottom_type',
        #'has_bottom_neighbor',
#count_neighbors_bottom	numeric
        #'bottom_neighbor_num',
        #'bottom_neighbor_str',
        #'bottom_neighbor_frml',
        #'bottom_neighbor_date',
        #'bottom_neighbor_aggr',
        #'bottom_neighbor_mix',
        #'matches_bottom_style',
        #'matches_left_type',
        #'has_left_neighbor',
#count_neighbors_left	numeric
        #'left_neighbor_num',
        #'left_neighbor_str',
        #'left_neighbor_frml',
        #'left_neighbor_date',
        #'left_neighbor_aggr',
        #'left_neighbor_mix',
        #'matches_left_style',
        #'matches_right_type',
        #'has_right_neighbor',
#count_neighbors_right	numeric
        #'right_neighbor_num',
        #'right_neighbor_str',
        #'right_neighbor_frml',
        #'right_neighbor_date',
        #'right_neighbor_aggr',
        #'right_neighbor_mix',
        #'matches_right_style',
        #'in_top_region',
        #'in_bottom_region',
        #'in_left_region',
        #'in_right_region',
    ]
    
    label_key = 'label'
    
    to_num_label = {
        "header": 0,
        "metadata": 1,
        "data": 2,
        "attributes": 3,
        "derived": 4
    }
    
    return {"X": np.array(list(map(lambda x: cell[x], feature_keys))), "Y": to_num_label[cell[label_key]]}
    
    
def create_grid(raw_grid):
    # initialize grid
    min_row = raw_grid['first_row_num'].min()
    max_row = raw_grid['last_row_num'].max()
    min_col = raw_grid['first_col_num'].min()
    max_col = raw_grid['last_col_num'].max()
    
    grid = [[None for i in range(max_row + 1)] for i in range(max_col + 1)]
    print((min_row, min_col, max_row, max_col))
    
    for index, row in raw_grid.iterrows():
        for i in range(row['first_col_num'],row['last_col_num'] + 1):
            for j in range(row['first_row_num'],row['last_row_num'] + 1):
                grid[i][j] = parse_featurevector(row)
                #print(i, row['last_col_num'] , row['first_col_num'])

        
    return grid
        
def create_sheets(df_cells, group, df):

    sheets = df_cells.groupby(['sheet_index']).groups
    grids = [create_grid(df.iloc[sheets[s]]) for i,s in enumerate(sheets)]
    return group, grids


processed_grid_corpus = [create_sheets(df.iloc[grouped_df[g]], g, df) for i, g in enumerate(grouped_df)]


for file, grid in processed_grid_corpus:
    print("{} -> {}".format(file, len(grid)))
    print(grid)
    
print(len(processed_grid_corpus))

# Save the dataset to pickle-file
output = open('data.pkl', 'wb')

# Pickle dictionary using protocol 0.
pickle.dump(processed_grid_corpus, output)
output.close()

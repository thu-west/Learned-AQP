ATTR_NUM = 10
MODEL_SAVE_PATH = "aqp.pt"

# example for train sets, it can be simply csv with n rows and 11 columns if we have 10 attributes
# the col0 represents the label, that is the percentage of estimated numbers among the total numbers.
# The estimated number is the result of query using the value of remaining columns
# attr1 < col1 and attr2 < col2 and .. and attr10 < col10 ( already normalized )
EG_TRAIN_SETS = [[0.49525818,
                  0.01514395, 0.12494134, 0.40093043, 0.12643129, 0.36664239,
                  0.75005143, 0.34194994, 0.88841597, 0.99224193, 0.93947453],
                 [0.94077041,
                  0.45948594, 0.93628831, 0.48432316, 0.02393118, 0.91079297,
                  0.18779871, 0.18345988, 0.27410979, 0.33953795, 0.98787811],
                 [0.84991375,
                  0.90906428, 0.55077781, 0.23517086, 0.23968738, 0.63598329,
                  0.89604262, 0.70026385, 0.28861198, 0.26619911, 0.89603062],
                 [0.98950196,
                  0.80230591, 0.07412405, 0.54295207, 0.04665015, 0.94063716,
                  0.51236693, 0.27361754, 0.40452523, 0.13658047, 0.78911306],
                 [0.22801487,
                  0.50269836, 0.93341963, 0.84558201, 0.87518842, 0.92128984,
                  0.46417814, 0.40452293, 0.20322377, 0.05345115, 0.55812737]]

import hsfs
connection = hsfs.connection()
fs = connection.get_feature_store(name='seta1_featurestore')
fg = fs.get_feature_group('credit_fg', version=1)
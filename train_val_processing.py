import pandas as pd
import sklearn.model_selection as model_selection

pageview = pd.read_csv('raw_data/pageview.csv', error_bad_lines=False)
content = pd.read_csv('raw_data/content.csv', error_bad_lines=False)
census = pd.read_csv('raw_data/census_data.csv', error_bad_lines=False)

#preprocess Census
census['north_south'] = census['NORTH_SOUTH'] == 'NORTH'
census['timezone'] = np.where(census['TIMEZONE'] == 'America/NewYork', 0,
                              np.where(census['TIMEZONE'] == 'America/Chicago', 1, 
                                  np.where(census['TIMEZONE'] == 'America/Los Angeles', 2,
                                           3)))
census = census.drop(['NORTH_SOUTH', 'TIMEZONE'], axis=1)
census_key_orig = census['CENSUS_KEY']
for col in census.columns:
    census[col] = pd.to_numeric(census[col], errors='coerce')
census['CENSUS_KEY'] = census_key_orig

#preprocess pageview
pageview = pageview.reindex(index=pageview.index[::-1])
pageview['cleaned_url'] = pageview['URL_PATH'].str.replace('/en', '')
pageview['cleaned_url'] = pageview['cleaned_url'].str.replace('/es-mx', '')

#preprocess Content
content.rename(columns={"url":"cleaned_url"},inplace=True)

#merge into a super_dataframe
pv_content_df = pd.merge(pageview,content,how='left',on='cleaned_url')
super_df = pd.merge(pv_content_df,census,how='left',on='CENSUS_KEY')


def train_val_split(dataframe=super_df,train_size=.8,test_size=.2):
    
    if dataframe= None:
        dataframe = pd.read_csv('raw_data/pageview.csv', error_bad_lines=False)
        pageview = pageview.reindex(index=pageview.index[::-1])

    else:
        pageview = pageview.reindex(index=pageview.index[::-1])

    super_df_train,super_df_test = \
        model_selection.train_test_split(super_df,train_size=.8,test_size=.2,random_state=None,shuffle=False)    
    return x_train, x_test, y_train, y_test


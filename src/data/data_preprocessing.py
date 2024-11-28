import pandas as pd
import numpy as np

MCI_PATH = './raw/major_crime_indicators.csv'
NEIGHBOURHOOD_PATH = './raw/neighbourhood_profiles.csv'
OUTPUT_PATH = 'toronto_crime_data.csv'


def _load_mci_df(path):
    df = pd.read_csv(path)
    df = df[(df['REPORT_YEAR'] >= 2021) & (df['REPORT_YEAR'] <= 2023) & \
            (df['HOOD_158'] != 'NSA')]
    
    df = df[['REPORT_YEAR', 'REPORT_MONTH', 'REPORT_DAY',
             'REPORT_DOY', 'REPORT_DOW', 'REPORT_HOUR', 
             'PREMISES_TYPE', 'OFFENCE', 
             'HOOD_158', 'LONG_WGS84', 'LAT_WGS84']]
    
    df = df.dropna()

    month_mapping = {
        'January': 1, 'February': 2, 'March': 3, 'April': 4,
        'May': 5, 'June': 6, 'July': 7, 'August': 8,
        'September': 9, 'October': 10, 'November': 11, 'December': 12
    }

    weekday_mapping = {
        'Monday': 1, 'Tuesday': 2, 'Wednesday': 3, 'Thursday': 4,
        'Friday': 5, 'Saturday': 6, 'Sunday': 7
    }

    df['REPORT_MONTH'] = df['REPORT_MONTH'].str.strip().str.capitalize()
    df['REPORT_DOW'] = df['REPORT_DOW'].str.strip().str.capitalize()
    df['REPORT_MONTH'] = df['REPORT_MONTH'].map(month_mapping)
    df['REPORT_DOW'] = df['REPORT_DOW'].map(weekday_mapping)
    
    df['HOOD_158'] = df['HOOD_158'].astype(int)
    df['OFFENCE'] = df['OFFENCE'].apply(_categorize_offence)
    df.rename(columns={'LONG_WGS84': 'LONGITUDE', 'LAT_WGS84': 'LATITUDE'}, inplace=True)

    for col in ['PREMISES_TYPE', 'OFFENCE']:
        df[col] = df[col].astype(str)

    for col in ['LONGITUDE', 'LATITUDE']:
        df[col] = df[col].astype(float)

    return df


def _load_neighbourhood_df(path):
    df = pd.read_csv(path)
    df = df[['Neighbourhood Number', 'TSNS 2020 Designation', 
                 'Total - Age groups of the population - 25% sample data', \
                 'Average age of the population', \
                 'Total - Income statistics in 2020 for the population aged 15 years and over in private households - 25% sample data', \
                 'Employment rate']]
    
    df.columns = ['HOOD_158', 'NBH_DESIGNATION', 'POPULATION', 'AVG_AGE', 'INCOME', 'EMPLOYMENT_RATE']

    df['NBH_DESIGNATION'] = df['NBH_DESIGNATION'].apply(lambda x: 0 if x == 'Not an NIA or Emerging Neighbourhood' else x)
    df['NBH_DESIGNATION'] = df['NBH_DESIGNATION'].apply(lambda x: 1 if x == 'Neighbourhood Improvement Area' else x)
    df['NBH_DESIGNATION'] = df['NBH_DESIGNATION'].apply(lambda x: 2 if x == 'Emerging Neighbourhood' else x)

    df = df.dropna()

    for col in ['AVG_AGE', 'INCOME', 'EMPLOYMENT_RATE']:
        df[col] = df[col].astype(float)

    for col in ['HOOD_158', 'POPULATION']:
        df[col] = df[col].astype(int)

    return df


def _categorize_offence(offence):
    high_risk = [
        'Administering Noxious Thing', 'Aggravated Aslt Peace Officer', 'Aggravated Assault',
        'Aggravated Assault Avails Pros', 'Air Gun Or Pistol: Bodily Harm', 'Assault Bodily Harm',
        'Assault Peace Officer', 'Assault Peace Officer Wpn/Cbh', 'Assault With Weapon',
        'Crim Negligence Bodily Harm', 'Disarming Peace/Public Officer', 
        'Discharge Firearm - Recklessly', 'Discharge Firearm With Intent', 
        'Hoax Terrorism Causing Bodily', 'Pointing A Firearm', 
        'Robbery - Armoured Car', 'Robbery - Home Invasion', 'Robbery With Weapon',
        'Robbery - Vehicle Jacking', 'Set/Place Trap/Intend Death/Bh', 
        'Traps Likely Cause Bodily Harm', 'Use Firearm / Immit Commit Off'
    ]
    
    if offence in high_risk:
        return 'High-risk'
    else:
        return 'Low-risk'


def _save_df(mci_path, neighbourhood_path, output_path):
    mci_df = _load_mci_df(mci_path)
    neighbourhood_df = _load_neighbourhood_df(neighbourhood_path)
    
    df = pd.merge(mci_df, neighbourhood_df, on='HOOD_158', how='right')
    df.to_csv(output_path, index=False)


if __name__ == '__main__':
    _save_df(MCI_PATH, NEIGHBOURHOOD_PATH, OUTPUT_PATH)


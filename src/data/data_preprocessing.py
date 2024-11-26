import pandas as pd


def _load_mci_df(path):
    df = pd.read_csv(path)
    df = df[(df['REPORT_YEAR'] >= 2021) & (df['REPORT_YEAR'] <= 2023) & \
                # empty entries for HOOD_158
                (df['HOOD_158'] != 'NSA')]
    
    df = df[['REPORT_DATE', 'REPORT_DOY', 'REPORT_DOW', \
                 'REPORT_HOUR', 'PREMISES_TYPE', 'OFFENCE', 'HOOD_158']]
    
    df['HOOD_158'] = df['HOOD_158'].astype(int)
    df['OFFENCE'] = df['OFFENCE'].apply(_categorize_offence)
    return df


def _load_neighbourhood_df(path):
    df = pd.read_csv(path)
    df = df[['Neighbourhood Number', 'TSNS 2020 Designation', 
                 'Total - Age groups of the population - 25% sample data', \
                 'Average age of the population', \
                 'Total - Income statistics in 2020 for the population aged 15 years and over in private households - 25% sample data', \
                 'Employment rate']]
    df.columns = ['HOOD_158', 'NBH_DESIGNATION', 'AVG_AGE', 'POPULATION', 'INCOME', 'EMPLOYMENT_RATE']

    df['NBH_DESIGNATION'] = df['NBH_DESIGNATION'].apply(lambda x: 0 if x == 'Not an NIA or Emerging Neighbourhood' else x)
    df['NBH_DESIGNATION'] = df['NBH_DESIGNATION'].apply(lambda x: 1 if x == 'Neighbourhood Improvement Area' else x)
    df['NBH_DESIGNATION'] = df['NBH_DESIGNATION'].apply(lambda x: 2 if x == 'Emerging Neighbourhood' else x)

    return df


def _categorize_offence(offence):
    if offence in [
        'Aggravated Assault', 'Aggravated Assault Avails Pros', 'Assault', 
        'Assault - Force/Thrt/Impede', 'Assault - Resist/Prevent Seiz', 
        'Assault Bodily Harm', 'Assault With Weapon', 
        'Crim Negligence Bodily Harm', 'Unlawfully Causing Bodily Harm'
    ]:
        return 'Assault and Violence Against Individuals'
    elif offence in [
        'Aggravated Aslt Peace Officer', 'Disarming Peace/Public Officer',
        'Assault Peace Officer', 'Assault Peace Officer Wpn/Cbh'
    ]:
        return 'Violence Against Authority Figures'
    elif offence in [
        'Air Gun Or Pistol: Bodily Harm', 'Pointing A Firearm', 
        'Discharge Firearm - Recklessly', 'Discharge Firearm With Intent', 
        'Use Firearm / Immit Commit Off'
    ]:
        return 'Firearms and Dangerous Weapons'
    elif offence in [
        'Robbery - Armoured Car', 'Robbery - Atm', 'Robbery - Business', 
        'Robbery - Delivery Person', 'Robbery - Financial Institute', 
        'Robbery - Home Invasion', 'Robbery - Mugging', 'Robbery - Other', 
        'Robbery - Purse Snatch', 'Robbery - Swarming', 'Robbery - Taxi', 
        'Robbery - Vehicle Jacking', 'Robbery To Steal Firearm', 
        'Robbery With Weapon'
    ]:
        return 'Robbery'
    elif offence in [
        'B&E', 'B&E - To Steal Firearm', 'B&E Out', "B&E W'Intent"
    ]:
        return 'Breaking and Entering'
    elif offence in [
        'Theft From Mail / Bag / Key', 'Theft From Motor Vehicle Over', 
        'Theft Of Motor Vehicle', 'Theft Of Utilities Over', 'Theft Over', 
        'Theft Over - Bicycle', 'Theft Over - Distraction', 'Theft Over - Shoplifting',
        'Theft - Misapprop Funds Over'
    ]:
        return 'Property Theft'
    elif offence in [
        'Administering Noxious Thing', 'Set/Place Trap/Intend Death/Bh', 
        'Traps Likely Cause Bodily Harm', 'Hoax Terrorism Causing Bodily'
    ]:
        return 'Crimes Causing Bodily Harm or Death'
    elif offence == 'Unlawfully In Dwelling-House':
        return 'Home Invasion and Trespassing'


def _save_df(mci_path, neighbourhood_path, output_path):
    mci_df = _load_mci_df(mci_path)
    neighbourhood_df = _load_neighbourhood_df(neighbourhood_path)
    
    df = pd.merge(mci_df, neighbourhood_df, on='HOOD_158', how='right')
    df.to_csv(output_path, index=False)


if __name__ == '__main__':
    mci_path = './raw/major_crime_indicators.csv'
    neighbourhood_path = './raw/neighbourhood_profiles.csv'
    output_path = 'toronto_crime_data.csv'

    _save_df(mci_path, neighbourhood_path, output_path)

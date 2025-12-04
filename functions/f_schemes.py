groups = {
    'North and Central America': ['GIC', 'NWN', 'NEN', 'WNA', 'CNA', 'ENA', 'NCA', 'SCA', 'CAR'],
    'South America': ['NWS', 'NSA', 'NES', 'SAM', 'SWS', 'SES', 'SSA'],
    'Europe': ['NEU', 'WCE', 'EEU', 'MED'],
    'Africa': ['SAH', 'WAF', 'CAF', 'NEAF', 'SEAF', 'WSAF', 'ESAF', 'MDG'],
    'Asia': ['RAR', 'WSB', 'ESB', 'RFE', 'WCA', 'ECA', 'TIB', 'EAS', 'ARP', 'SAS'],
    'Australasia': ['SEA', 'NAU', 'CAU', 'EAU', 'SAU', 'NZ'],
    'Antarctica': ['EAN', 'WAN'],
}

continent_abbrevs = ['NA', 'SA', 'EU', 'AF', 'Asia', 'AU', 'ANT']


def generate_single_region_constraint_pairs(groups: dict) -> list:
    """
    Generate region-level constraint pairs for emergent constraint analysis.

    Each target region is constrained using all regions in its continental group.

    Parameters:
    -----------
    groups : dict
        Dictionary where keys are group/continent names and values are lists of region abbreviations.

    Returns:
    --------
    List[dict]
        A list of dictionaries with keys:
        - 'target_reg': individual region code
        - 'constrain_used_reg': list of region codes from the same group
    """
    pairs = []
    for group_regions in groups.values():
        for region in group_regions:
            pairs.append({
                'target_reg': region,
                'constrain_used_reg': region
            })
    return pairs

def generate_continent_allregion_constraint_pairs(groups: dict) -> list:
    """
    Generate region-level constraint pairs for emergent constraint analysis.

    Each target region is constrained using all regions in its continental group.

    Parameters:
    -----------
    groups : dict
        Dictionary where keys are group/continent names and values are lists of region abbreviations.

    Returns:
    --------
    List[dict]
        A list of dictionaries with keys:
        - 'target_reg': individual region code
        - 'constrain_used_reg': list of region codes from the same group
    """
    pairs = []
    for group_regions in groups.values():
        for region in group_regions:
            pairs.append({
                'target_reg': region,
                'constrain_used_reg': group_regions
            })
    return pairs



def generate_continent_mean_constraint_pairs(groups: dict, continent_abbrevs: list) -> list:
    """
    Generate constraint pairs where each region is constrained by its continent abbreviation.

    Parameters:
    -----------
    groups : dict
        Dictionary mapping continent names to lists of region abbreviations.
    continent_abbrevs : list
        List of continent abbreviations in the same order as the groups.

    Returns:
    --------
    List[dict]
        A list of dictionaries with:
        - 'target_reg': individual region abbreviation
        - 'constrain_used_reg': corresponding continent abbreviation
    """
    constraint_pairs = []
    for abbrev, (group_name, region_list) in zip(continent_abbrevs, groups.items()):
        for region in region_list:
            constraint_pairs.append({
                'target_reg': region,
                'constrain_used_reg': abbrev
            })
    return constraint_pairs


def generate_global_target_constraint_pairs(groups, global_abbrev='GLB'):
    pairs = []
    for region_list in groups.values():
        for region in region_list:
            pairs.append({
                'target_reg': region,
                'constrain_used_reg': global_abbrev
            })
    return pairs


def generate_global_regional_target_constraint_pairs(groups, global_abbrev='GLB'):
    pairs = []
    for region_list in groups.values():
        for region in region_list:
            pairs.append({
                'target_reg': region,
                'constrain_used_reg': [region, global_abbrev]
            })

    # for entry in pairs:
    #     entry['constrain_used_reg'] = list(dict.fromkeys(entry['constrain_used_reg']))
        
    return pairs


def generate_continent_and_global_constraint_pairs(groups, continent_abbrevs, global_abbrev='GLB'):
    """
    Generate target-constrain pairs where each target region is constrained by
    its continent abbrev and global abbrev.

    Parameters:
        groups (dict): Mapping of continent names to list of region abbrevs.
        continent_abbrevs (list): Abbreviations for each continent, in the same order as `groups`.
        global_abbrev (str): Global constraint label (default: 'GLB').

    Returns:
        list of dicts with 'target_reg' and 'constrain_used_reg' = [continent_abbrev, 'GLB']
    """
    constraint_pairs = []
    for abbrev, (group_name, region_list) in zip(continent_abbrevs, groups.items()):
        for region in region_list:
            constraint_pairs.append({
                'target_reg': region,
                'constrain_used_reg': [abbrev, global_abbrev]
            })

    # for entry in constraint_pairs:
    #     entry['constrain_used_reg'] = list(dict.fromkeys(entry['constrain_used_reg']))
        
    return constraint_pairs


def generate_global_north_target_constraint_pairs(groups, north_abbrev='north', global_abbrev='LSAT'):
    """
    For each target region, generate a dict with:
        - target_reg: the region itself
        - constrain_used_reg: list containing [region, north_abbrev, global_abbrev]
    """
    pairs = []
    for region_list in groups.values():
        for region in region_list:
            pairs.append({
                'target_reg': region,
                'constrain_used_reg': [region, north_abbrev, global_abbrev]
            })
    return pairs
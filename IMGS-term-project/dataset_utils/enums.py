class Enums:
    
    mapping_dict = {
        'Car': 'Car',
        'Van': 'Car',
        'Truck': 'Car',
        'Pedestrian': 'Pedestrian',
        'Person_sitting': 'Pedestrian',
        'Cyclist': 'Cyclist'
    }
    
    KiTTi_label2Id = {
        'Car': 0,
        'Cyclist': 1,
        'Pedestrian': 2
    }

    KiTTi_Id2label = {
        0: 'Car',
        1: 'Cyclist',
        2: 'Pedestrian',
    }

    reduced_KiTTi_label2Id = {
        'Car': 0,
        'Van': 0,
        'Truck': 0,
        'Pedestrian': 3,
        'Person_sitting': 3,
        'Cyclist': 1,
        'Tram': 2,
        'Misc': 2,
        'DontCare': 2
    }

    # KiTTi_label2Id = {
    #     'Car': 0,
    #     'Van': 1,
    #     'Truck': 2,
    #     'Pedestrian': 3,
    #     'Person_sitting': 4,
    #     'Cyclist': 5,
    #     'Tram': 6,
    #     'Misc': 7,
    #     'DontCare': 8
    # } 
    
    # KiTTi_Id2label = {
    #     0: 'Car',
    #     1: 'Van',
    #     2: 'Truck',
    #     3: 'Pedestrian',
    #     4: 'Person_sitting',
    #     5: 'Cyclist',
    #     6: 'Tram',
    #     7: 'Misc',
    #     8: 'DontCare'
    # }
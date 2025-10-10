table_info = [
    {
        'name': 'geo_12_users',
        'description': 'Table listing MissionOS users',
        'columns': [
            {'name': 'id', 'description': 'unique identifier for user'},
            {'name': 'username', 'description': 'user name for login e.g. john_smith'},
            {'name': 'name', 'description': 'name of user for salutation e.g. John'},
            {'name': 'user_type', 'description': 'id referencing the id column of table mg_user_types'}
        ],
        'relationships': [
            {'column': 'user_type', 'referenced_table': 'mg_user_types', 'referenced_column': 'id'}
        ]
    },
    {
        'name': 'mg_user_types',
        'description': 'Table listing types of user, conferring create, read, update and delete permissions',
        'columns': [
            {'name': 'id', 'description': 'unique identifier for user type'},
            {'name': 'name', 'description': 'name of user type'},
            {'name': 'status', 'description': '1 -> active, 0 -> inactive'}
        ],
        'relationships': []
    },
    {
        'name': 'user_access_groups_users',
        'description': 'Table assigning users to hierarchy access groups',
        'columns': [
            {'name': 'id', 'description': 'unique identifier for a user-to-group assignment'},
            {'name': 'group_id', 'description': 'id referencing the id column of table user_access_groups'},
            {'name': 'user_id', 'description': 'id referencing the id column of table geo_12_users'},
            {'name': 'permission', 'description': 'set of strings describing what a user is allowed to do'},
            {'name': 'user_deleted', 'description': '1 -> user deleted, 0 -> user not deleted'}
        ],
        'relationships': [
            {'column': 'group_id', 'referenced_table': 'user_access_groups', 'referenced_column': 'id'},
            {'column': 'user_id', 'referenced_table': 'geo_12_users', 'referenced_column': 'id'}
        ]
    },
    {
        'name': 'user_access_groups_permissions',
        'description': 'Table defining permissions defining hierarchy accessible by groups or individual users',
        'columns': [
            {'name': 'id', 'description': 'unique identifier for permission'},
            {'name': 'group_id', 'description': 'GROUP -> assigned to group, USER -> assigned to individual user'},
            {'name': 'user_group_id', 'description': 'id referencing id column of table user_access_groups'},
            {'name': 'project', 'description': 'id referencing id column of table projects for project accessible under permission, 0 if all projects accessible'},
            {'name': 'contract', 'description': 'id referencing id column of table contracts for contract accessible under permission, 0 if all contracts accessible'},
            {'name': 'site', 'description': 'id referencing id column of table sites for site accessible under permission, 0 if all sites accessible'}
        ],
        'relationships': [
            {'column': 'user_group_id', 'referenced_table': 'user_access_groups', 'referenced_column': 'id'},
            {'column': 'project', 'referenced_table': 'projects', 'referenced_column': 'id'},
            {'column': 'contract', 'referenced_table': 'contracts', 'referenced_column': 'id'},
            {'column': 'site', 'referenced_table': 'sites', 'referenced_column': 'id'}
        ]
    },
    {
        'name': 'projects',
        'description': 'Table listing projects set-up in MissionOS',
        'columns': [
            {'name': 'id', 'description': 'unique identifier for project'},
            {'name': 'name', 'description': 'project name'},
            {'name': 'description', 'description': 'project description'},
            {'name': 'is_deleted', 'description': '1 -> deleted, 0 -> not deleted'}
        ],
        'relationships': []
    },
    {
        'name': 'contracts',
        'description': 'Table listing contracts set-up in MissionOS',
        'columns': [
            {'name': 'id', 'description': 'unique identifier for contract'},
            {'name': 'name', 'description': 'contract name'},
            {'name': 'project_id', 'description': 'id of project that contract belongs to. References the id column of table projects'},
            {'name': 'is_deleted', 'description': '1 -> deleted, 0 -> not deleted'}
        ],
        'relationships': [
            {'column': 'project_id', 'referenced_table': 'projects', 'referenced_column': 'id'}
        ]
    },
    {
        'name': 'sites',
        'description': 'Table listing sites set-up in MissionOS',
        'columns': [
            {'name': 'id', 'description': 'unique identifier for site'},
            {'name': 'name', 'description': 'site name'},
            {'name': 'contract_id', 'description': 'id of contract that site belongs to. References the id column of table contracts'},
            {'name': 'is_deleted', 'description': '1 -> deleted, 0 -> not deleted'}
        ],
        'relationships': [
            {'column': 'contract_id', 'referenced_table': 'contracts', 'referenced_column': 'id'}
        ]
    },
    {
        'name': 'zones',
        'description': 'Table listing zones set-up in MissionOS',
        'columns': [
            {'name': 'id', 'description': 'unique identifier for zone'},
            {'name': 'name', 'description': 'zone name'},
            {'name': 'site_id', 'description': 'id of site that zone belongs to. References the id column of table sites'},
            {'name': 'is_deleted', 'description': '1 -> deleted, 0 -> not deleted'}
        ],
        'relationships': [
            {'column': 'site_id', 'referenced_table': 'sites', 'referenced_column': 'id'}
        ]
    },
    {
        'name': 'location',
        'description': 'Table listing locations of instruments in terms of eastings and northings',
        'columns': [
            {'name': 'id', 'description': 'unique identifier for location'},
            {'name': 'name', 'description': 'name of instrument, usually corresponding to instrument ID'},
            {'name': 'easting', 'description': 'easting of instrument according to local coordinate system'},
            {'name': 'northing', 'description': 'northing of instrument according to local coordinate system'},
            {'name': 'project_id', 'description': 'id of project under which instrument falls. References column id of table projects'},
            {'name': 'contract_id', 'description': 'id of contract under which instrument falls. References column id of table contracts'},
            {'name': 'site_id', 'description': 'id of site under which instrument falls. References column id of table sites'},
            {'name': 'zone_id', 'description': 'id of zone under which instrument falls. References column id of table zones'}
        ],
        'relationships': [
            {'column': 'project_id', 'referenced_table': 'projects', 'referenced_column': 'id'},
            {'column': 'contract_id', 'referenced_table': 'contracts', 'referenced_column': 'id'},
            {'column': 'site_id', 'referenced_table': 'sites', 'referenced_column': 'id'},
            {'column': 'zone_id', 'referenced_table': 'zones', 'referenced_column': 'id'}
        ],
        'hierarchy_permissions_implementation': {
            'num_joins_from_location_table': 0,
            'sql_extension_template': """
                WITH original_query AS ({original_query}) 
                SELECT original_query.* 
                FROM original_query 
                JOIN location ON location.id = original_query.id 
                WHERE original_query.project_id NOT IN ({project_ids}) 
                AND original_query.contract_id NOT IN ({contract_ids}) 
                AND original_query.site_id NOT IN ({site_ids})
            """
        }
    },
    {
        'name': 'instrum',
        'description': 'Table listing instruments. Use the object_ID column to reference table type_config_normalized for \
            context on instrument fields and hence the instrument type. DO NOT rely on the type1 and subtype1 columns as these are \
            labels for system categorisation only.',
        'columns': [
            {'name': 'id', 'description': 'unique identifier for instrument'},
            {'name': 'object_ID', 'description': 'composite identifier for type1 and subtype1 columns'},
            {'name': 'type1', 'description': 'instrument type. Type is a way to categorise instruments. \
             This column typically contains abbreviations e.g. VWP for vibrating wire piezometer.'},
            {'name': 'subtype1', 'description': 'instrument subtype. Subtype is a subcategory of type'},
            {'name': 'instr_id', 'description': 'instrument id, typically synonymous with the instrument name'},
            {'name': 'instr_level', 'description': 'elevation of instrument'},
            {'name': 'location_id', 'description': 'id referencing column id of table location'},
            {'name': 'date_installed', 'description': 'installation date of instrument'}
        ],
        'relationships': [
            {'column': 'location_id', 'referenced_table': 'location', 'referenced_column': 'id'}
        ]
    },
    {
        'name': 'hierarchy_members',
        'description': 'Table assigning child instruments to parent (master) instruments. \
         The word "hierarchy" used in this table differs totally from the project-contract-site-zone hierarchy.',
        'columns': [
            {'name': 'id', 'description': 'unique identifier for a parent-child assignment'},
            {'name': 'hierarchy_id', 'description': 'id referencing column id of table hierarchies. Look-up in the table hierarchies will give the parent instrument in the assignment'},
            {'name': 'instr_id', 'description': 'id of the child instrument in the assignment. References column instr_id of table instrum'}
        ],
        'relationships': [
            {'column': 'hierarchy_id', 'referenced_table': 'hierarchies', 'referenced_column': 'id'},
            {'column': 'instr_id', 'referenced_table': 'instrum', 'referenced_column': 'instr_id'}
        ]
    },
    {
        'name': 'hierarchies',
        'description': 'Table listing parent (master) instruments. \
            The "hierarchy" naming used in this table differs totally from the project-contract-site-zone hierarchy.',
        'columns': [
            {'name': 'id', 'description': 'unique identifier for parent instrument'},
            {'name': 'master_instr', 'description': 'instrument id for parent instrument. References column instr_id of table instrum'},
            {'name': 'is_derived', 'description': 'describes whether parent instrument is derived or composite 1 -> derived, 0 -> composite'}
        ],
        'relationships': [
            {'column': 'master_instr', 'referenced_table': 'instrum', 'referenced_column': 'instr_id'}
        ]
    },
    {
        'name': 'mydata',
        'description': 'Table listing approved uploaded time-series readings from instruments. \
         Calculated fields are processed values calculated from uploaded readings. \
         Calculated fields are named sequentially calculation1, calculation2 etc. \
         ALWAYS reference table type_config_normalized to get the context for system field names data1, data2, ... data12. \
         ALWAYS reference column `custom_fields` to get values for calculated fields. \
         Table type_config_normalized can be referenced via table instrum using instr_id and object_ID columns.',
        'columns': [
            {'name': 'instr_id', 'description': 'id of instrument where reading was taken. References column instr_id of table instrum'},
            {'name': 'date1', 'description': 'timestamp of reading'},
            {'name': 'id', 'description': 'unique identifier for reading'},
            {'name': 'data1', 'description': 'reading value corresponding to system field name data1'},
            {'name': 'data2', 'description': 'reading value corresponding to system field name data2'},
            {'name': 'data3', 'description': 'reading value corresponding to system field name data3'},
            {'name': 'data4', 'description': 'reading value corresponding to system field name data4'},
            {'name': 'data5', 'description': 'reading value corresponding to system field name data5'},
            {'name': 'data6', 'description': 'reading value corresponding to system field name data6'},
            {'name': 'data7', 'description': 'reading value corresponding to system field name data7'},
            {'name': 'data8', 'description': 'reading value corresponding to system field name data8'},
            {'name': 'data9', 'description': 'reading value corresponding to system field name data9'},
            {'name': 'data10', 'description': 'reading value corresponding to system field name data10'},
            {'name': 'data11', 'description': 'reading value corresponding to system field name data11'},
            {'name': 'data12', 'description': 'reading value corresponding to system field name data12'},
            {'name': 'remarks', 'description': 'comments about the reading uploaded with the reading'},
            {'name': 'custom_fields', 'description': 'JSON string defining values of calculated fields'}
        ],
        'relationships': [
            {'column': 'instr_id', 'referenced_table': 'instrum', 'referenced_column': 'instr_id'}
        ]
    },
    {
        'name': 'instr_cal_calibs',
        'description': 'Table listing calculated calibration fields for instruments \
         Calculated calibration fields are computed via a function, whilst calibration fields are inputted directly by users.',
        'columns': [
            {'name': 'id', 'description': 'unique identifier of field'},
            {'name': 'instr_id', 'description': 'id of instrument to which field pertains. References column instr_id of table instrum'},
            {'name': 'hierarchy_id', 'description': 'id of parent-child assignment, if the instrument is a child. If instrument is not a child, defaults to 0. References column id of table hierarchies'},
            {'name': 'master_instr', 'description': 'id of parent instrument, if the instrument is a child. References column instr_id of table instrum'},
            {'name': 'calc_cali_field', 'description': 'system field name of field'},
            {'name': 'calc_cali_value', 'description': 'value of field'},
            {'name': 'created_on', 'description': 'timestamp when field was created'},
            {'name': 'updated_on', 'description': 'timestamp when field was updated'}
        ],
        'relationships': [
            {'column': 'instr_id', 'referenced_table': 'instrum', 'referenced_column': 'instr_id'},
            {'column': 'hierarchy_id', 'referenced_table': 'hierarchies', 'referenced_column': 'id'},
            {'column': 'master_instr', 'referenced_table': 'instrum', 'referenced_column': 'instr_id'}
        ]
    },
    {
        'name': 'review_instruments',
        'description': 'Table listing reviews set-up on fields. \
            A review is a check on whether the value of a field exceeds or drops below specified threshold values. \
            This table lists reviews on both instruments and construction jobs.',
        'columns': [
            {'name': 'id', 'description': 'unique identifier of review'},
            {'name': 'item_id', 'description': '0 -> review on instrument. If review on job, then this will be the job id'},
            {'name': 'instr_id', 'description': 'if review on instrument, id of instrument that review is set-up on. References column instr_id of table instrum. if review on job, id of position for job'},
            {'name': 'review_type', 'description': '1 -> upper (checks if field value is greater than threshold), -1 -> lower (checks if field value is lower than threshold), 0 -> upper and lower'},
            {'name': 'review_field', 'description': 'system field name for field review is set-up on'},
            {'name': 'review_status', 'description': 'ON -> review active, OFF -> review inactive'},
            {'name': 'effective_from', 'description': 'timestamp when review becomes effective'},
            {'name': 'effective_to', 'description': 'timestamp when review stops being effective'},
            {'name': 'created_on', 'description': 'timestamp when review was created'}
        ],
        'relationships': [
            {'column': 'instr_id', 'referenced_table': 'instrum', 'referenced_column': 'instr_id'}
        ]
    },
    {
        'name': 'review_instruments_values',
        'description': 'Table listing threshold values for each review. \
         A review typically comprises multiple review levels. \
         Each review level has a threshold value.',
        'columns': [
            {'name': 'id', 'description': 'unique identifier for review level of a review'},
            {'name': 'review_instr_id', 'description': 'references column id of table review_instruments'},
            {'name': 'review_level_id', 'description': 'indicates review level within a review. References column id of table review_levels'},
            {'name': 'review_direction', 'description': '1 -> upper (checks if field value is greater than threshold), -1 -> lower (checks if field value is lower than threshold)'},
            {'name': 'review_value', 'description': 'value of threshold for review level'},
            {'name': 'is_breached', 'description': '1 -> breached, 0 -> not breached'}
        ],
        'relationships': [
            {'column': 'review_instr_id', 'referenced_table': 'review_instruments', 'referenced_column': 'id'},
            {'column': 'review_level_id', 'referenced_table': 'review_levels', 'referenced_column': 'id'}
        ]
    },
    {
        'name': 'review_levels',
        'description': 'Table listing types of review level which can comprise a review',
        'columns': [
            {'name': 'id', 'description': 'unique identifier for review level type'},
            {'name': 'review_name', 'description': 'name of review level type'},
            {'name': 'is_deleted', 'description': '1 -> deleted, 0 -> not deleted'}
        ],
        'relationships': []
    },
    {
        'name': 'type_config_normalized',
        'description': 'Table giving information on instrument fields including naming, formulae and units',
        'columns': [
            {'name': 'config_id', 'description': 'unique identifier for an instrument field'},
            {'name': 'object_ID', 'description': 'composite identifier for type and subtype columns. References column object_ID of table instrum'},
            {'name': 'type', 'description': 'instrument type. Type is a way to categorise instruments. References column type1 of table instrum'},
            {'name': 'subtype', 'description': 'instrument subtype. Subtype is a subcategory of type. References column subtype1 of table instrum'},
            {'name': 'field_name', 'description': 'system field name for field: \
             taken_on for timestamp of reading, \
             data1, data2, data3, ... for uploaded time-series readings, \
             cali1, cali2, cali3, ... for calculated calibration data calculated from uploaded calibration data, \
             cal_cali1, calc_cali2, calc_cali3, ... for uploaded calibration data, \
             calculation1, calculation2, calculation3, ... for time-series metrics calculated from uploaded readings and calibration data, \
             remarks for uploaded comments on reading'},
            {'name': 'field_type', 'description': 'type of field: \
             data: uploaded time-series readings, \
             cali: uploaded or calculated calibration data defining instrument set-up, \
             calc: time-series metrics calculated from uploaded readings and calibration data'},
            {'name': 'user_field_name', 'description': 'user-defined name for field'},
            {'name': 'user_label', 'description': 'user-defined label for field e.g. for graph axis labelling'},
            {'name': 'user_description', 'description': 'user-defined description for field'},
            {'name': 'db_formula', 'description': 'formula for field with field names mapped to database columns. \
             Fields are stated as <table_name>.<column_name> format. \
             Table name "rd" is synonymous with "mydata".'},
            {'name': 'user_formula', 'description': 'formula for field with field names as system field names'},
            {'name': 'units', 'description': 'units of measurement for field'},
            {'name': 'precision_value', 'description': 'number of decimal places to store field value'},
            {'name': 'calibration_label', 'description': 'user-defined label for fields of type "cali" e.g. for graph axis labelling'},
            {'name': 'calibration_desc', 'description': 'user-defined description for fields of type "cali"'},
            {'name': 'calibration_field_name', 'description': 'user-defined name for fields of type "cali"'},
            {'name': 'display_order', 'description': ' \
             for fields of type "calc": order of calculation of field, \
             for other field types: order of display of fields e.g. in uploaded tables'},
            {'name': 'created_on', 'description': 'timestamp when field was created'},
            {'name': 'updated_on', 'description': 'timestamp when field was updated'}
        ]
    },
    {
        'name': 'types',
        'description': 'Table listing instrument types their descriptions. \
         Use this table to understand the context of instrument types.',
        'columns': [
            {'name': 'id', 'description': 'unique identifier for table rows'},
            {'name': 'type1', 'description': 'instrument type. Type is a way to categorise instruments. \
             References column type1 of table instrum'},
            {'name': 'descript', 'description': 'description of instrument type. \
             Use this to understand the context of instrument type.'},
            {'name': 'subtype1', 'description': 'instrument subtype. Subtype is a subcategory of type'}
        ]
    },
    {
        'name': 'graph_template',
        'description': 'Table defining graph labelling for plotting instrument fields. \
         Use this table to understand the context of instrument fields.',
        'columns': [
            {'name': 'id', 'description': 'unique identifier for a graph'},
            {'name': 'graph_name', 'description': 'name of graph in graph collection'},
            {'name': 'type', 'description': 'type of instrument whose fields are plotted in the graph. \
             If this instrument is compound, the type will be that of the parent instrument. \
             References column type of table type_config_normalized and column type1 of table instrum'},
            {'name': 'subtype', 'description': 'subtype of instrument whose fields are plotted in the graph. \
             If this instrument is compound, the subtype will be that of the parent instrument. \
             References column subtype of table type_config_normalized and column subtype1 of table instrum'},
            {'name': 'child_type', 'description': 'type of child instrument whose fields are plotted in the graph. \
             Only relevant to compound instruments. \
             References column type of table type_config_normalized and column type1 of table instrum'},
            {'name': 'child_sub_type', 'description': 'subtype of child instrument whose fields are plotted in the graph. \
             Only relevant to compound instruments. \
             References column subtype of table type_config_normalized and column subtype1 of table instrum'},
            {'name': 'Graph_title1', 'description': 'title of graph. \
             Use this to understand the context of instrument fields plotted in the graph'},
            {'name': 'x', 'description': 'system field name for field plotted on x-axis of graph. \
             References column field_name of table type_config_normalized'},
            {'name': 'y', 'description': 'system field name for field plotted on y-axis of graph. \
             References column field_name of table type_config_normalized'},
            {'name': 'x_caption', 'description': 'x-axis label for graph. \
             Use this to understand the context of instrument field plotted on x-axis of graph'},
            {'name': 'y_caption', 'description': 'y-axis label for graph. \
             Use this to understand the context of instrument field plotted on y-axis of graph'}
        ]
    }
]
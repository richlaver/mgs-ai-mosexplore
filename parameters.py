from langchain_core.messages import AIMessage
platform_context = [
    {
        'name': 'review_levels',
        'query_keywords': [
            'review', 'exceedance', 'threshold', 'level', 'status', 'breach', 'exceed', 'trigger', 'AAA', 'alert', 'action', 'alarm', 'limit'
        ],
        'context': """
# Review Levels
Readings are compared with thresholds called **review levels** to indicate unexpected or unsafe behaviour.
Other names for *review* include **AAA** (abbreviation of Alert, Action, Alarm) and **trigger**.
Other names for *level* include **threshold**.
When a reading surpasses a review level, it is called a review level **exceedance**, **breach** or similar.
In a set of review levels, each level represents a degree of severity in exceedance.
The name of the level which a reading is said to exceed is the most severe level that has been surpassed by the reading.
This name is also referred to as **review status** or **exceedance level**.
For example if a set comprises three progressive levels called *alert*, *action* and *alarm* in order of increasing severity, a reading is said to:
- “exceed the *alert* level” (review status: *alert*) if the reading lies between the *alert* and *action* levels
- “exceed the *action* level” (review status: *action*) if the reading lies between the *action* and *alarm* levels
- “exceed the *alarm* level” (review status: *alarm*) if the reading surpasses the *alarm* level.
A reading not surpassing any review level has a review status of **not exceeded**, **not breached** or similar.
A set of review levels can be either **upper** or **lower**.
An upper review level is surpassed if a reading is **greater than** the review level.
A lower review level is surpassed if a reading **less than** the review level.
A reading field can be assigned with upper review levels, lower review levels, both or none.
Database table `review_instruments` lists reading fields assigned with review levels:
- The name of the reading field in the database is in column `review_field` e.g. “data1”, “calculation1”
- The instrument ID to which the reading field belongs is in column `instr_id`
- To filter by instrument type and/or subtype: join column `instr_id` to {{table: `instrum`, column: `instr_id`}} and reference {{table: `instrum`, columns: [`type1`, `subtype1`]}}
- To filter by location coordinates: join column `instr_id` to {{table: `instrum`, column: `instr_id`}} then join {{table: `instrum`, column: `location_id`}} to {{table: `location`, column: `id`}} and reference {{table: `location`, columns: [`easting`, `northing`]}}
- To filter by project, contract, site or zone: join column `instr_id` to {{table: `instrum`, column: `instr_id`}} then join {{table: `instrum`, column: `location_id`}} to {{table: `location`, column: `id`}} and reference {{table: `location`, columns: [`project_id`, `contract_id`, `site_id`, `zone_id`]}}
The value for each review level is listed in database table `review_instruments_values`:
- Column `review_direction` indicates whether the level is *upper* or *lower*
- Column `review_value` gives the value of the level
- Column `review_instr_id` joins to {{table: `review_instruments`, column: `id`}}
- Review status can be evaluated by using table joins to compare readings in table `mydata` to the `review_value` column
The name for each review level is listed in the database in {{table: `review_levels`, column: `review_name`}}. Column `id` joins to {{table: `review_instruments_values`, column: `review_level_id`}}
"""
    }
]
progress_messages = {
    'context_orchestrator_node': AIMessage(
        name="ContextOrchestrator",
        content="Gathering context on query...",
        additional_kwargs={
            "stage": "node",
            "process": "context_orchestrator"
        }
    ),
    'history_summariser_node': AIMessage(
        name="HistorySummariser",
        content="Summarising the chat history...",
        additional_kwargs={
            "stage": "node",
            "process": "history_summariser"
        }
    ),
    'execution_initializer_node': AIMessage(
        name="ExecutionInitializer",
        content="Preparing CodeAct executors to answer your query...",
        additional_kwargs={
            "stage": "node",
            "process": "execution_initializer"
        }
    ),
    'enter_parallel_execution_node': AIMessage(
        name="EnterParallelExecutionNode",
        content="Implementing a strategy to answer your query...",
        additional_kwargs={
            "stage": "node",
            "process": "enter_parallel_execution_node"
        }
    ),
    'reporter_node': AIMessage(
        name="Reporter",
        content="Drafting my response to your query...",
        additional_kwargs={
            "stage": "node",
            "process": "reporter"
        }
    )
}
users = [
    {
        'id': 1,
        'display_name': 'Super Admin' 
    },
    {
        'id': 200,
        'display_name': 'User (Osvaldo Moro)'
    },
    {
        'id': 26,
        'display_name': 'MGS Developer (Charuthu)' 
    },
    {
        'id': 185,
        'display_name': 'User (Tran Nguyen Quan)' 
    },
    {
        'id': 7777777,
        'display_name': 'Not-Existent User' 
    },
]
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
        'description': 'Table listing instruments.',
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
         Calculated fields are named sequentially calculation1, calculation2 etc.',
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
             If this instrument is compound, the type will be that of the parent instrument.'},
            {'name': 'subtype', 'description': 'subtype of instrument whose fields are plotted in the graph. \
             If this instrument is compound, the subtype will be that of the parent instrument.'},
            {'name': 'child_type', 'description': 'type of child instrument whose fields are plotted in the graph. \
             Only relevant to compound instruments.'},
            {'name': 'child_sub_type', 'description': 'subtype of child instrument whose fields are plotted in the graph. \
             Only relevant to compound instruments.'},
            {'name': 'Graph_title1', 'description': 'title of graph. \
             Use this to understand the context of instrument fields plotted in the graph'},
            {'name': 'x', 'description': 'system field name for field plotted on x-axis of graph.'},
            {'name': 'y', 'description': 'system field name for field plotted on y-axis of graph.'},
            {'name': 'x_caption', 'description': 'x-axis label for graph. \
             Use this to understand the context of instrument field plotted on x-axis of graph'},
            {'name': 'y_caption', 'description': 'y-axis label for graph. \
             Use this to understand the context of instrument field plotted on y-axis of graph'}
        ]
    }
]
include_tables = [table['name'] for table in table_info]
custom_table_info = {
    "geo_12_users": """
        /*
        Table listing MissionOS users.
        - id: unique identifier for user
        - username: user name for login e.g. john_smith
        - name: name of user for salutation e.g. John
        - user_type: id referencing the id column of table mg_user_types
        */
        CREATE TABLE "geo_12_users" (
            "id" INT UNSIGNED AUTO_INCREMENT PRIMARY KEY NOT NULL,
            "username" VARCHAR(255) NOT NULL,
            "name" VARCHAR(100) NOT NULL,
            "user_type" INT REFERENCES "mg_user_types" ("id")
        );
        SELECT "id", "username", "name", "user_type" FROM "geo_12_users" LIMIT 3;
    """,
    "mg_user_types": """
        /*
        Table listing types of user, confering create, read, update and delete permissions.
        - id: unique identifier for user type
        - name: name of user type
        - status: 1 -> active, 0 -> inactive
        */
        CREATE TABLE "mg_user_types" (
            "id" INT UNSIGNED PRIMARY KEY NOT NULL,
            "name" VARCHAR(255) UNIQUE NOT NULL,
            "status" TINYINT DEFAULT 1 NOT NULL
        );
        SELECT "id", "name", "status" FROM "mg_user_types" LIMIT 3;
    """,
    'user_access_groups_users': """
        /*
        Table assigning users to hierarchy access groups.
        - id: unique identifier for a user-to-group assignment
        - group_id: id referencing the id column of table user_access_groups
        - user_id: id referencing the id column of table geo_12_users
        - permission: set of strings describing what a user is allowed to do
        - user_deleted: 1 -> user deleted, 0 -> user not deleted
        */
        CREATE TABLE "user_access_groups_users" (
            "id" INT NOT NULL AUTO_INCREMENT PRIMARY KEY,
            "group_id" INT REFERENCES "user_access_groups" ("id"),
            "user_id" INT REFERENCES "geo_12_users" ("id"),
            "permission" SET('APPROVAL','SEE','REMOVE','MODIFY','ADD','UPLOAD_DATA','CREATE_INSTRUMENT','GET_ALARM','GET_ACTION','GET_ALERT','GET_NOTIF'),
            "user_deleted": INT DEFAULT 0
        );
        SELECT "id", "group_id", "user_id", "permission", "user_deleted" FROM "user_access_groups_users" LIMIT 3;
    """,
    'user_access_groups_permissions': """
        /*
        Table defining permissions defining hierarchy accessible by groups or individual users. 
        Hierarchy is nested with levels from top to bottom: project -> contract -> site -> zone.
        Permissions are only set for accessing project, contract and site.
        All zones within a site are accessible if a site is accessible.
        - id: unique identifier for permission
        - access_type: GROUP -> assigned to group, USER -> assigned to individual user
        - user_group_id: id referencing id column of table user_access_groups
        - project: id referencing id column of table projects for project accessible under permission, 0 if all projects accessible
        - contract: id referencing id column of table contracts for contract accessible under permission, 0 if all contracts accessible
        - site: id referencing id column of table sites for site accessible under permission, 0 if all sites accessible
        */
        CREATE TABLE "user_access_groups_permissions" (
            "id" INT NOT NULL AUTO_INCREMENT PRIMARY KEY,
            "group_id" ENUM('USER','GROUP') NOT NULL,
            "user_group_id" INT NOT NULL REFERENCES "user_access_groups" ("id"),
            "project" INT NOT NULL REFERENCES "projects" ("id"),
            "contract" INT NOT NULL REFERENCES "contracts" ("id"),
            "site" INT NOT NULL REFERENCES "sites" ("id")
        );
        SELECT "id", "group_id", "user_group_id", "project", "contract", "site" FROM "user_access_groups_permissions" LIMIT 3;
    """,
    'projects': """
        /*
        Table listing projects set-up in MissionOS.
        - id: unique identifier for project
        - name: project name
        - description: project description
        - is_deleted: 1 -> deleted, 0 -> not deleted
        */
        CREATE TABLE "projects" (
            "id" INT NOT NULL AUTO_INCREMENT PRIMARY KEY,
            "name" VARCHAR(50) NOT NULL,
            "description" TEXT NOT NULL,
            "is_deleted" INT NOT NULL DEFAULT 0
        );
        SELECT "id", "name", "description", "is_deleted" FROM "projects" LIMIT 3;
    """,
    'contracts': """
        /*
        Table listing contracts set-up in MissionOS.
        - id: unique identifier for contract
        - name: contract name
        - project_id: id of project that contract belongs to. References the id column of table projects.
        - is_deleted: 1 -> deleted, 0 -> not deleted
        */
        CREATE TABLE "contracts" (
            "id" INT NOT NULL AUTO_INCREMENT PRIMARY KEY,
            "name" VARCHAR(50) NOT NULL,
            "project_id" INT NOT NULL REFERENCES "projects" ("id"),
            "is_deleted" INT NOT NULL DEFAULT 0
        );
        SELECT "id", "name", "project_id", "is_deleted" FROM "contracts" LIMIT 3;
    """,
    'sites': """
        /*
        Table listing sites set-up in MissionOS.
        - id: unique identifier for site
        - name: site name
        - contract_id: id of contract that site belongs to. References the id column of table contracts.
        - is_deleted: 1 -> deleted, 0 -> not deleted
        */
        CREATE TABLE "sites" (
            "id" INT NOT NULL AUTO_INCREMENT PRIMARY KEY,
            "name" VARCHAR(50) NOT NULL,
            "contract_id" INT NOT NULL REFERENCES "contracts" ("id"),
            "is_deleted" INT NOT NULL DEFAULT 0
        );
        SELECT "id", "name", "contract_id", "is_deleted" FROM "sites" LIMIT 3;
    """,
    'zones': """
        /*
        Table listing zones set-up in MissionOS.
        - id: unique identifier for zone
        - name: zone name
        - site_id: id of site that zone belongs to. References the id column of table sites.
        - is_deleted: 1 -> deleted, 0 -> not deleted
        */
        CREATE TABLE "zones" (
            "id" INT NOT NULL AUTO_INCREMENT PRIMARY KEY,
            "name" VARCHAR(50) NOT NULL,
            "site_id" INT NOT NULL REFERENCES "sites" ("id"),
            "is_deleted" INT NOT NULL DEFAULT 0
        );
        SELECT "id", "name", "site_id", "is_deleted" FROM "zones" LIMIT 3;
    """,
    'location': """
        /*
        Table listing locations of instruments in terms of eastings and northings. Look here for instrument coordinates.
        - id: unique identifier for location
        - name: name of instrument, usually corresponding to instrument ID
        - easting: easting of instrument according to local coordinate system
        - northing: northing of instrument according to local coordinate system
        - project_id: id of project under which instrument falls. References column id of table projects.
        - contract_id: id of contract under which instrument falls. References column id of table contracts.
        - site_id: id of site under which instrument falls. References column id of table sites.
        - zone_id: id of zone under which instrument falls. References column id of table zones.
        */
        CREATE TABLE "location" (
            "id" INT NOT NULL AUTO_INCREMENT PRIMARY KEY,
            "name" VARCHAR(50) UNIQUE,
            "easting" VARCHAR(20),
            "northing" VARCHAR(20),
            "project_id" INT NOT NULL REFERENCES "projects" ("id"),
            "contract_id" INT NOT NULL REFERENCES "contracts" ("id"),
            "site_id" INT NOT NULL REFERENCES "sites" ("id"),
            "zone_id" INT NOT NULL REFERENCES "zones" ("id")
        );
        SELECT "id", "name", "easting", "northing", "project_id", "contract_id", "site_id", "zone_id" FROM "location" LIMIT 3;
    """,
    'instrum': """
        /*
        Table listing instruments.
        - id: unique identifier for instrument
        - object_ID: composite identifier for type1 and subtype1 columns
        - type1: instrument type. Type is a way to categorise instruments.
        - subtype1: instrument subtype. Subtype is a subcategory of type.
        - instr_id: instrument id, typically synonymous with the instrument name
        - instr_level: elevation of instrument
        - location_id: id referencing column id of table location
        - date_installed: installation date of instrument
        */
        CREATE TABLE "instrum" (
            "id" INT UNSIGNED NOT NULL AUTO_INCREMENT PRIMARY KEY,
            "object_ID" INT NOT NULL,
            "type1" VARCHAR(25),
            "subtype1" VARCHAR(50) DEFAULT 'DEFAULT',
            "instr_id" VARCHAR(50),
            "instr_level" VARCHAR(50),
            "location_id" INT REFERENCES "location" ("id"),
            "date_installed" DATETIME
        );
        SELECT "id", "object_ID", "type1", "subtype1", "instr_id", "instr_level", "location_id", "date_installed" FROM "instrum" LIMIT 3;
    """,
    'hierarchy_members': """
        /*
        Table assigning child instruments to parent (master) instruments.
        The word "hierarchy" used in this table differs totally from the project-contract-site-zone hierarchy.
        - id: unique identifier for a parent-child assignment
        - hierarchy_id: id referencing column id of table hierarchies. Look-up in the table hierarchies will give the parent instrument in the assignment.
        - instr_id: id of the child instrument in the assignment. References column instr_id of table instrum.
        */
        CREATE TABLE "hierarchy_members" (
            "id" INT UNSIGNED NOT NULL AUTO_INCREMENT PRIMARY KEY,
            "hierarchy_id" INT UNSIGNED NOT NULL REFERENCES "hierarchies" ("id"),
            "instr_id" VARCHAR(50) NOT NULL REFERENCES "instrum" ("instr_id")
        );
        SELECT "id", "hierarchy_id", "instr_id" FROM "hierarchy_members" LIMIT 3;
    """,
    'hierarchies': """
        /*
        Table listing parent (master) instruments.
        The "hierarchy" naming used in this table differs totally from the project-contract-site-zone hierarchy.
        - id: unique identifier for parent instrument
        - master_instr: instrument id for parent instrument. References column instr_id of table instrum.
        - is_derived: describes whether parent instrument is derived or composite 1 -> derived, 0 -> composite.
        */
        CREATE TABLE "hierarchies" (
            "id" INT UNSIGNED NOT NULL AUTO_INCREMENT PRIMARY KEY,
            "master_instr" VARCHAR(50),
            "is_derived" TINYINT DEFAULT 0
        );
        SELECT "id", "master_instr", "is_derived" FROM "hierarchies" LIMIT 3;
    """,
    'mydata': """
        /*
        Table listing approved uploaded readings from instruments.
        - instr_id: id of instrument where reading was taken. References column instr_id of table instrum.
        - date1: timestamp of reading
        - id: unique identifier for reading
        - data1: reading value corresponding to system field name data1.
        Other reading values are stored analogously:
        - datan: reading value corresponding to system field name datan, where n is an integer differentiating between fields.
        - remarks: comments about the reading uploaded with the reading
        - custom_fields: JSON string defining values of calculated fields: { "<system_field_name>": <field_value>, ... }
        Calculated fields are processed values calculated from uploaded readings.
        Calculated fields are named sequentially calculation1, calculation2 etc.
        */
        CREATE TABLE "mydata" (
            "instr_id" VARCHAR(50) REFERENCES "instrum" ("instr_id"),
            "date1" DATETIME,
            "id" INT NOT NULL AUTO_INCREMENT,
            "data1" VARCHAR(50),
            "data2" VARCHAR(50),
            "data3" VARCHAR(50),
            "data4" VARCHAR(50),
            "data5" VARCHAR(50),
            "data6" VARCHAR(50),
            "data7" VARCHAR(50),
            "data8" VARCHAR(50),
            "data9" VARCHAR(50),
            "data10" VARCHAR(50),
            "data11" VARCHAR(50),
            "data12" VARCHAR(50),
            "remarks" VARCHAR(255),
            "custom_fields" MEDIUMTEXT
        );
        SELECT "instr_id", "date1", "id", "data1", "data2", "data3", "data4", "data5", "data6", "data7", "data8", "data9", "data10", "data11", "data12", "remarks", "custom_fields" FROM "mydata" LIMIT 3;
    """,
    'instr_cal_calibs': """
        /*
        Table listing calculated calibration fields for instruments.
        Calculated calibration fields are computed via a function, whilst calibration fields are inputted directly by users.
        - id: unique identifier of field
        - instr_id: id of instrument to which field pertains. References column instr_id of table instrum.
        - hierarchy_id: id of parent-child assignment, if the instrument is a child. If instrument is not a child, defaults to 0. References column id of table hierarchies.
        - master_instr: id of parent instrument, if the instrument is a child. References column instr_id of table instrum.
        - calc_cali_field: system field name of field
        - calc_cali_value: value of field
        - created_on: timestamp when field was created
        - updated_on: timestamp when field was updated
        */
        CREATE TABLE "instr_cal_calibs" (
            "id" INT NOT NULL AUTO_INCREMENT PRIMARY KEY,
            "instr_id" VARCHAR(50) REFERENCES "instrum" ("instr_id"),
            "hierarchy_id" INT NOT NULL DEFAULT 0 REFERENCES "hierarchies" ("id"),
            "master_instr" VARCHAR(50) NOT NULL,
            "calc_cali_field" VARCHAR(250) NOT NULL,
            "calc_cali_value" DOUBLE NOT NULL,
            "created_on" DATETIME NOT NULL,
            "updated_on" DATETIME NOT NULL
            ""
        );
        SELECT "id", "instr_id", "hierarchy_id", "master_instr", "calc_cali_field", "calc_cali_value", "created_on", "updated_on" FROM "instr_cal_calibs" LIMIT 3;
    """,
    'review_instruments': """
        /*
        Table listing reviews set-up on fields.
        A review is a check on whether the value of a field exceeds or drops below specified threshold values.
        This table lists reviews on both instruments and construction jobs.
        - id: unique identifier of review
        - item_id: 0 -> review on instrument. If review on job, then this will be the job id.
        - instr_id: if review on instrument, id of instrument that review is set-up on. References column instr_id of table instrum.
            if review on job, id of position for job.
        - review_type:
            1 -> upper (checks if field value is greater than threshold)
            -1 -> lower (checks if field value is lower than threshold)
            0 -> upper and lower
        - review_field: system field name for field review is set-up on
        - review_status: ON -> review active, OFF -> review inactive
        - effective_from: timestamp when review becomes effective
        - effective_to: timestamp when review stops being effective
        - created_on: timestamp when review was created
        */
        CREATE TABLE "review_instruments" (
            "id" INT NOT NULL AUTO_INCREMENT PRIMARY KEY,
            "item_id" INT NOT NULL DEFAULT 0,
            "instr_id" VARCHAR(256) NOT NULL REFERENCES "instrum" ("instr_id"),
            "review_type" ENUM("0","1","-1"),
            "review_field" VARCHAR(256) NOT NULL,
            "review_status" ENUM('ON','OFF'),
            "effective_from" DATETIME NOT NULL,
            "effective_to" DATETIME NOT NULL,
            "created_on" DATETIME NOT NULL
        );
        SELECT "id", "item_id", "instr_id", "review_field", "review_status", "effective_from", "effective_to", "created_on" FROM "review_instruments" LIMIT 3;
    """,
    'review_instruments_values': """
        /*
        Table listing threshold values for each review.
        A review typically comprises multiple review levels.
        Each review level has a threshold value.
        - id: unique identifier for review level of a review
        - review_instr_id: references column id of table review_instruments
        - review_level_id: indicates review level within a review. References column id of table review_levels
        - review_direction: 1 -> upper (checks if field value is greater than threshold), -1 -> lower (checks if field value is lower than threshold)
        - review_value: value of threshold for review level
        - is_breached: 1 -> breached, 0 -> not breached
        */
        CREATE TABLE "review_instruments_values" (
            "id" INT NOT NULL AUTO_INCREMENT PRIMARY KEY,
            "review_instr_id" INT NOT NULL REFERENCES "review_instruments" ("id"),
            "review_level_id" INT NOT NULL REFERENCES "review_levels" ("id"),
            "review_direction" TINYINT NOT NULL,
            "review_value" VARCHAR(50),
            "is_breached" INT NOT NULL
        );
        SELECT "id", "review_instr_id", "review_level_id", "review_direction", "review_value", "is_breached" FROM "review_instruments_values" LIMIT 3;
    """,
    'review_levels': """
        /*
        Table listing types of review level which can comprise a review.
        - id: unique identifier for review level type
        - review_name: name of review level type
        - is_deleted: 1 -> deleted, 0 -> not deleted
        */
        CREATE TABLE "review_levels" (
            "id" INT NOT NULL AUTO_INCREMENT PRIMARY KEY,
            "review_name" VARCHAR(50) NOT NULL,
            "is_deleted" TINYINT(1) NOT NULL DEFAULT 0
        );
        SELECT "id", "review_name", "is_deleted" FROM "review_levels" LIMIT 3;
    """
}
trend_context = [
    {
        'instrument_types_subtypes': [
            {
                'type': 'LP',
                'subtypes': ['DEFAULT']
            }
        ],
        'measurands': ['settlement', 'vertical displacement'],
        'behaviour': [
            {
                'mode': 'fluctuating',
                'causes': ['random survey error'],
                'impacts': ['inaccurate measurement'],
                'notes': 'Survey errors typically have an amplitude consistent for a particular survey method. Survey errors will be persistent across all time periods. Error amplitude can be up to 5 mm.'
            },
            {
                'mode': 'steady',
                'causes': [],
                'impacts': [],
                'notes': 'Be aware that changes in between readings may be missed.'
            },
            {
                'mode': 'gradual fall',
                'causes': [
                    'consolidation due to groundwater drawdown',
                    'nearby excavation',
                    'nearby construction activity causing ground compaction',
                    'ground losses due to tunnelling',
                    'slow landslide'
                ],
                'impacts': [
                    'damage to adjacent utilities, structures or buildings',
                    'uneven ground surface e.g. for road, railtrack or runway'
                ],
                'notes': 'Consolidation is marked by a progressively decreasing settlement rate, although the decrease may not be obvious over short time periods. The rate of settlement due to nearby excavation is likely to vary depending on excavation progress. Landslide movements should accelerate with higher groundwater levels e.g. due to rainfall.'
            },
            {
                'mode': 'gradual rise',
                'causes': [
                    'nearby grout injection',
                    'swelling of clay soils',
                    'upward displacement at the base of a slip circle',
                    'over-pressure at face of Earth Pressure Balance (EPB) shield'
                ],
                'impacts': [
                    'damage to adjacent utilities, structures or buildings',
                    'uneven ground surface e.g. for road, railtrack or runway'
                ],
                'notes': 'Swelling causes very small heave rates. Heave due to grouting and tunnelling is typically more pronounced.'
            },
            {
                'mode': 'sudden fall',
                'causes': [
                    'sinkhole formation or collapse of underground voids',
                    'sudden surcharge loading',
                    'nearby compaction activity',
                    'accidental disturbance of marker'
                ],
                'impacts': [
                    'damage to adjacent utilities, structures or buildings',
                    'risk of further sudden settlements',
                    'potholes and major surface cracking'
                ],
                'notes': 'Void collapses can occur in quick succession, separated by steady phases.'
            },
            {
                'mode': 'sudden rise',
                'causes': [
                    'railtrack tamping',
                    'accidental disturbance of marker'
                ],
                'impacts': [],
                'notes': 'Sudden rises are uncommon and are usually benign.'
            },
            {
                'mode': 'spike',
                'causes': [
                    'error in survey, data processing or data entry'
                ],
                'impacts': [],
                'notes': 'An error is usually characterised by an anomalous single reading. Readings return to normal after the error.'
            }
        ]
    }
]
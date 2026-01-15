from langchain_core.messages import AIMessage
platform_context = [
    {
        'name': 'review_levels',
        'query_keywords': [
            'review', 'exceedance', 'threshold', 'level', 'status', 'breach', 'exceed', 'trigger', 'AAA', 'alert', 'action', 'alarm', 'limit', 'elapsed', 'missing'
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
    },
    {
        'name': 'settlement_polarity',
        'query_keywords': [
            'settlement', 'settle', 'heave', 'subsidence', 'subside', 'level', 'elevation', 'vertical'
        ],
        'context': """
# Sign Convention for Settlement
Assume that negative settlement readings represent settlement (downward movement) and positive settlement readings represent heave (upward movement) unless otherwise clarified. 
"""
    },
    {
        'name': 'compound_instruments',
        'query_keywords': [
            'inclinometer', 'extensometer', 'shape accel array', 'saa', 'liquid level', 'hydrostatic levelling cell', 'hlc', 'distributed fibre optic strain sensing', 'dfoss', 'compound', 'master', 'parent', 'child'
        ],
        'context': """
# Compound Instruments
An instrument comprising multiple sensors is called a **compound** instrument.
In such an instrument, sensors are typically distributed along a line.
Examples include:
- Inclinometer
- Extensometer
- Shape Accel Array (SAA)
- Liquid levelling system or hydrostatic levelling cell (HLC)
- Distributed Fibre Optic Strain Sensing (DFOSS)
- System estimating deflection by integrating the measured tilt of connected rigid segments with distance
In the database the association of sensors with the instrument is represented by a **parent-child relationship**:
- Each sensor is represented by a **child** instrument.
- The sensors collectively belong to a **parent** instrument.
The query will generally refer to the instrument ID of the parent instrument when referring to the collection of sensors as a whole.
The parent instrument itself is *not* a physical sensor but is an entity to represent and name its collection of child sensors.
Readings are only recorded at child instruments; nonetheless parent instruments typically have calculation fields storing values representative of the collection of sensors as a whole.
Database tables `hierarchies` and `hierarchy_members` describe parent-child relationships for compound instruments:
- To find child instruments belonging to a parent instrument, filter {{table: `hierarchies`, column: `master_instr`}} by the parent instrument ID, join {{table: `hierarchies`, column: `id`}} to {{table: `hierarchy_members`, column: `hierarchy_id`}} and extract {{table: `hierarchy_members`, column: `instr_id`}}
Infer the relative position of child instruments from:
- Instrument level: for vertically-aligned sensors (e.g. inclinometers, extensometers) relative position may be indicated by instrument elevation as recorded in {{table: `instrum`, column: `instr_level`}}
- Instrument order: {{table: `hierarchy_members`, column: `relation`}} stores an integer index describing the relative ordering of sensors. Referencing {{table: `hierarchies`, column: `collection_order`}} for the parent instrument gives the order direction as `ASC` (ascending) or `DESC` (descending)
- Instrument ID: often suffixed with an index describing relative order, depth, elevation or chainage
Find the location of sensors by referencing {{table: `location`}} filtering {{column: `name`}} for the parent instrument ID.
Child instruments are typically *not* listed in {{table: `location`}}. 
"""
    },
    {
        'name': 'derived_instruments',
        'query_keywords': [
            'derived', 'tilt', 'differential'
        ],
        'context': """
# Derived Instruments
Instruments can be grouped together to enable new metrics to be calculated by combining readings from the different instruments.
A group of such instruments is called a **derived** instrument.
Examples of derived instruments include:
- Settlement markers between which tilt or differential settlement is calculated
- Two instruments measuring the same physical characteristic between which any discrepancy is calculated
- Prisms between which separation distance is calculated
In the database a special instrument represents the group in a **parent-child relationship**.
The special instrument is the parent whilst the children are the instruments within the group.
The parent instrument itself is *not* a physical instrument but is an entity to represent and name the group of child instruments.
Database tables `hierarchies` and `hierarchy_members` describe parent-child relationships for derived instruments:
- To find child instruments belonging to a parent instrument, filter {{table: `hierarchies`, column: `master_instr`}} by the parent instrument ID, join {{table: `hierarchies`, column: `id`}} to {{table: `hierarchy_members`, column: `hierarchy_id`}} and extract {{table: `hierarchy_members`, column: `instr_id`}}
A derived instrument with parent instrument ID in {{table: `hierarchies`, column: `master_instr`}} will have 1 and not 0 in {{table: `hierarchies`, column: `is_derived`}}.
Child instruments within a derived instrument can be of different types and subtypes. 
"""
    },
    {
        'name': 'inclinometers',
        'query_keywords': [
            'inclinometer', 'lateral', 'deflection', 'cumulative', 'incremental', 'resultant', 'A-face', 'B-face', 'A-direction', 'B-direction', 'A-axis', 'B-axis', 'profile'
        ],
        'context': """
# Inclinometers
Inclinometers measure subsurface lateral deflection either of the surrounding soil or an installed element like a retaining wall or pile.
Inclinometers comprise a vertical array of rigid segments connected end-to-end, each with a tilt sensor.
The base of the inclinometer is assumed to be installed deep enough undergo zero deflection and is hence adopted as a *datum*.
**Deflection** is computed by integrating measured sensor tilts with distance from the base upward.
The sensors are not installed perfectly vertical and so the inclinometer has an **initial deflection**.
The deflection measured at a particular time is therefore corrected by subtracting the initial deflection to obtain the **displacement**.
Displacement is typically reported as **two different metrics**:
- **Incremental displacement**: The difference in displacement between the top and bottom of a sensor, and represents *relative movement* between adjacent measurement points. Excessive incremental displacement (> 200mm) at a sensor or group of adjacent sensors (< 6) indicates shear zone formation implying failure of the surrounding medium (soil or structure)
- **Cumulative displacement**: The displacement of the top of a sensor relative to the base datum and represents *absolute movement*. Cumulative displacement is usually used to compare against review levels to ascertain whether movements are exceeding predictions or are onerous.
Both metrics are plotted as:
- Variation with time for individual sensors
- Variation with elevation at a particular time (displacement profile with elevation or depth)
The shape of the cumulative displacement profile reveals typical underlying causes:
- Small variations (< 5mm) across short distances (< 5m) but no persistent trend (< 10 mm) from bottom to top: random measurement error with no insignificant movement
- Gradually increasing displacement from bottom to top (> 10mm): adjacent unsupported excavation or global slope movement
- Displacement peaks (> 20mm) somewhere between bottom and top so profile represents a bulge: adjacent strutted excavation
- Step change (> 200mm) over a short distance (< 2.5m): highly localised shear zone and failure of surrounding medium e.g. slip plane in slope
Deflection and hence displacement is measured in two orthogonal lateral directions:
- **A-face**: Also called *A-direction* or *A-axis*. The positive A axis usually points towards the direction of interest e.g. towards an excavation or downhill on a slope. The inclinometer bearing will usually refer to the angle of the positive A axis from north measured clockwise when looking downward.
- **B-face**: Also called *B-direction* or *B-axis*. The positive B axis is 90 degrees from the positive A axis measured clockwise when looking downward.
Queries will typically require A-face displacement because this is measured in the direction of interest so **always** extract and use only the A-face displacement for answering queries unless the query specifies otherwise.
If the query requires **resultant** displacement then you must compute the resultant displacement vector from the A-face and B-face displacements.
B-face displacement is often susceptible to error due to how it is measured and *should not be extracted or used* unless specifically requested or if resultant displacement is required.
When you compute resultant displacement treat displacements as 2D **vectors** in a horizontal plane:
- First find the inclinometer bearing which will be stored either in a calibration field or a data field
- If a bearing is available then resolve A-face and B-face displacements as north and east displacements by rotating between coordinate systems, else continue with A-face and B-face displacements alone
- To find change in resultant displacement between start and end times subtract the resultant displacement vector at the end time from that at the start time
- Always state any resultant vector in your answer with both *magnitude* and *bearing*.
- If the inclinometer bearing was available state the resultant bearing with respect to both *north* and the *positive A-face*, otherwise state only with respect to the positive A-face
In the database, inclinometers are stored as *compound* instruments with each sensor as a child instrument of a parent instrument which represents the inclinometer as a whole.
Any instrument ID for an inclinometer in the query will usually refer to that of the *parent* instrument and not of any child.
Remember that when finding maximum displacement at an inclinometer you need to check *all* child instruments along the inclinometer.
Remember to state the *elevation* of any displacement you mention in your answer.
When extracting displacement values, **always** refer to *child* instruments not the parent instrument because only summary values are stored for the parent instrument; nonetheless for review level checks reference both parent and child instruments because review levels may be applied to the parent instrument, the child instruments or both.
Review levels will typically be applied to database fields storing *cumulative* displacement and not incremental displacement. 
"""
    },
    {
        'name': 'elapsed_time',
        'query_keywords': [
            'elapsed', 'overdue', 'missing'
        ],
        'context': """
# Elapsed Time
Readings at instruments are usually taken at regular time intervals.
When readings are missed there is a risk that critical issues may pass undetected and safety, progress or cost may be compromised.
Missed readings are also called **overdue** readings.
A special set of review levels called **elapsed time** review levels thus provides thresholds on the time elapsed since the most recent reading was taken so that relevant parties can be notified when a reading is overdue.
Like other review levels, elapsed time review levels can comprise progressive levels of severity, although usually only a single level is necessary and is defined with a value.
Instruments defined with an elapsed time review level are identified by the label “elapsed_time” in {{table: review_instruments, column: review_field}} for an instrument with instrument ID in {{table: review_instruments, column: instr_id}}.
Do **not** extract elapsed time review level data like other review levels because the data is stored differently.
Values for elapsed time review level thresholds are stored in {{table: review_instruments_values, column: review_value}} in units of seconds.
Exceedances of elapsed time review levels can be readily extracted from table `overdue_notif` with columns as follows:
- `instr_id`: instrument ID
- `review_id`: references {{table: review_instruments_values, column: id}}
- `review_level`: references {{table: review_levels, column: id}}
- `elapsed_value`: time elapsed since last reading
This table *only* lists instruments with elapsed time review level exceedance.
Elapsed time since last reading for *any* instrument can be deduced by finding the latest reading in table `mydata`.
To be a valid reading at least one `data`-type field must have a *numeric* value (not a string or empty).
In table `mydata`, `data`-type fields are stored in columns named `dataN`, where `N` is an integer.
Find the names of available `data`-type fields from the background behind words in the query.
The review direction for elapsed time review levels will *always* be 1 (upper) because elapsed time cannot be negative.
Always search for exceedances of elapsed time review levels whenever the query asks about missing or overdue readings. 
"""
    },
    {
        'name': 'instrument_model',
        'query_keywords': [
            'model', 'type', 'subtype', 'field', 'calibration', 'calculated', 'data', 'install'
        ],
        'context': """
# Instrument Model
In the database, every instrument belongs to a **type** and a **subtype**.
Subtype is a sub-category of type.
Type typically categorises instruments by function: *what* an instrument measures.
Subtype typically categorises instruments by method: *how* an instrument measures.
Subtype can therefore differentiate between manufacturer, brand or model.
The type and subtype are stored in {{table: instrum, columns: [type, subtype]}} for an instrument with instrument ID in {{table: instrum, column: instr_id}}.
For each subtype an **instrument model** is defined and describes what data is stored for that subtype and how it is processed.
Data is stored in **fields**.
A field can be one of four types:
- **Calibration** or `cal_cali`. Properties of an instrument that are time invariant e.g. fixed offset to apply to readings.
- **Data** or `data`. Time series instrument readings e.g. the elevation measured by surveying. The names of `data` fields in the database follow the pattern `dataN` where `N` is an integer. Data fields are stored in {{table: mydata, columns: [data1, data2, …]}}.
- **Calculation** or `calculation`. Time series quantities calculated from instrument readings e.g. settlement calculated from a measured elevation change. The names of `calculation` fields in the database follow the pattern `calculationN` where `N` is an integer. Calculation fields are stored in a JSON string in {{table: mydata, column: custom_fields}} as values of keys named `calculation1`, `calculation2`, … etc.
- **Calculated calibration** or `cali_calcs`. Properties of an instrument that depend on one or more of the `cali`, `data` and `calc` fields e.g. properties that vary with time e.g. time-dependent correction factor for temperature effect. The names of `cali_calcs` fields in the database follow the pattern `cali_calcsN` where `N` is an integer. Find calculated calibration values in {{table: instr_cal_calibs, column: calc_cali_value}} filtering by instrument ID and field name in {{table: instr_cal_calibs, columns: [instr_id, calc_cali_value]}}.
To access installation date or calibration fields of an instrument:
1. Find the name of the table where these are stored in {{table: raw_instr_typestbl, column: cal_table}} filtering by {{table: raw_instr_typestbl, columns: [type, subtype]}} for the type and subtype of the instrument.
2. Refer to relevant columns of the table are as follows:
a. `instr_id`: instrument ID
b. `date_installed`: installation date
c. `cal_caliN` (where `N` is an integer): calculated calibration field value 
"""
    },
    {
        'name': 'hierarchy',
        'query_keywords': [
            'project', 'contract', 'site', 'zone', 'hierarchy'
        ],
        'context': """
# Hierarchy
Each instrument is categorised within a **hierarchy** according to how the instrument is administered.
The hierarchy is structured as follows from top to bottom:
1. **Project**: A construction project financed by a distinct client e.g. bridge, tunnel, building. The name of the project is typically named by the client. Projects are stored in the table `projects`. A project will comprise one or more contracts.
2. **Contract**: The client may divide the project into different scopes of work or *contracts*, each of which will be awarded to a contractor to execute. Contracts are stored in the table `contracts`. The project to which a contract belongs is indicated by the project ID in {{table: contracts, column: project_id}}. A contract will comprise one or more sites.
3. **Site**: The contractor may divide the contract into different areas which are called *sites* to easily identify different portions of the work. Sites are stored in the table `sites`. The contract to which a site belongs is indicated by the contract ID in {{table: sites, column: contract_id}}. A site will comprise one or more zones.
4. **Zone**: The contractor may further divide the site into different zones to more precisely pinpoint a work area. Zones are stored in the table `zones`. The site to which a zone belongs is indicated by the site ID in {{table: zones, column: site_id}}.
If categorisation is sufficient at a particular level of hierarchy without the need to further categorise to lower levels then typically only one option called “Default” will be given for each level which is lower than that particular level.
When a query requests data for a particular project, contract, site or zone you will need to filter instruments accordingly.
User access permissions are often governed by this administration hierarchy and a user may be restricted from accessing instruments within a particular project, contract, site or zone. 
"""
    },
    {
        'name': 'status',
        'query_keywords': [
            'status', 'decommission', 'maintenance', 'obstruct', 'repair', 'fault', 'inactive', 'active', 'removed', 'offline', 'damaged', 'inaccessible', 'reinstated', 'lost'
        ],
        'context': """
# Instrument Status
Each instrument can be labeled with an **instrument status** indicating the current state that the instrument is in which is often used to suggest whether the instrument is fit to take readings and if not why.
The *instrument* status differs from the *review level* status.
Refer to instrument status to explain why readings are currently missing particularly over a prolonged period (> 7 days), or if the query specifically requests instrument status e.g. “How many instruments are decommissioned?”.
Find instrument status by referencing {{table: instrument_status_configuration, column: instrum_status}} and joining {{table: instrument_status_configuration, column: id}} to {{table: instrument_remarks, column: rem_status_id}} for an instrument with instrument ID in {{table: instrument_remarks, column: instr_id}}. 
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
        ],
        'relationships': [
            {'column': 'type1', 'referenced_table': 'instrum', 'referenced_column': 'type1'},
            {'column': 'subtype1', 'referenced_table': 'instrum', 'referenced_column': 'subtype1'}
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
        ],
        'relationships': [
            {'column': 'type', 'referenced_table': 'instrum', 'referenced_column': 'type1'},
            {'column': 'subtype', 'referenced_table': 'instrum', 'referenced_column': 'subtype1'},
            {'column': 'child_type', 'referenced_table': 'instrum', 'referenced_column': 'type1'},
            {'column': 'child_sub_type', 'referenced_table': 'instrum', 'referenced_column': 'subtype1'}
        ]
    },
    {
        'name': 'overdue_notif',
        'description': 'Table listing elapsed time review level exceedances. \
         Use this table to extract instruments where readings are overdue.',
        'columns': [
            {'name': 'id', 'description': 'unique identifier for an elapsed time exceedance'},
            {'name': 'instr_id', 'description': 'identifier of the instrument with overdue readings'},
            {'name': 'review_id', 'description': 'references column id of table review_instruments_values'},
            {'name': 'review_level', 'description': 'references column id of table review_levels'},
            {'name': 'elapsed_value', 'description': 'time elapsed since last reading in seconds'}
        ],
        'relationships': [
            {'column': 'instr_id', 'referenced_table': 'instrum', 'referenced_column': 'instr_id'},
            {'column': 'review_id', 'referenced_table': 'review_instruments_values', 'referenced_column': 'id'},
            {'column': 'review_level', 'referenced_table': 'review_levels', 'referenced_column': 'id'}
        ]
    },
    {
        'name': 'instrument_remarks',
        'description': 'Table indicating current status of instruments. \
         Use this table to clarify why instruments might currently have missing readings.',
        'columns': [
            {'name': 'rem_id', 'description': 'unique identifier for a status record'},
            {'name': 'rem_status_id', 'description': 'identifier indicating status of instrument. \
             References column id of table instrument_status_configuration.'},
            {'name': 'instr_id', 'description': 'identifier for instrument whose status is being indicated. \
             References column instr_id of table instrum.'},
            {'name': 'status_date', 'description': 'date when status took effect'}
        ],
        'relationships': [
            {'column': 'rem_status_id', 'referenced_table': 'instrument_status_configuration', 'referenced_column': 'id'},
            {'column': 'instr_id', 'referenced_table': 'instrum', 'referenced_column': 'instr_id'}
        ]
    },
    {
        'name': 'instrument_status_configuration',
        'description': 'Table listing possible instrument statuses. \
         Use this table to find the meaning of status identifiers in table instrument_remarks.',
        'columns': [
            {'name': 'id', 'description': 'unique identifier for a possible instrument status'},
            {'name': 'instrum_status', 'description': 'descriptive label for status'}
        ],
        'relationships': []
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
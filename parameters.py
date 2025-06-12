include_tables=[
    'geo_12_users',
    'mg_user_types',
    'user_access_groups_users',
    'user_access_groups',
    'user_access_groups_permissions',
    'projects',
    'contracts',
    'sites',
    'zones',
    'location',
    'instrum',
    'raw_instr_typestbl',
    # 'object_class_type',
    # 'object_class',
    'hierarchy_members',
    'hierarchies',
    'review_instruments',
    'review_instruments_values',
    'review_levels'
]
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
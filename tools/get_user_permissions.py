from typing import List, Optional, Type
from langchain_core.tools import BaseTool
from langchain_core.callbacks import CallbackManagerForToolRun
from pydantic import BaseModel, Field, ConfigDict
from langchain_community.utilities.sql_database import SQLDatabase


class BaseSQLDatabaseTool(BaseModel):
    """Base tool for interacting with a SQL database."""

    db: SQLDatabase = Field(exclude=True)

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
    )

class SiteDict(BaseModel):
    site_id: int = Field(description='Unique identifier of site within the contract')
    site_name: str = Field(description='Name of site within the contract')

class ContractDict(BaseModel):
    contract_id: int = Field(description='Unique identifier of contract within the project')
    contract_name: str = Field(description='Name of contract within the project')
    specific_sites: List[SiteDict] = Field(description="""
        List of specific sites within the contract the user can access. If the list is empty, the user can access ALL sites within the contract.
    """)

class ProjectDict(BaseModel):
    project_id: int = Field(description='Unique identifier of project')
    project_name: str = Field(description='Name of project')
    specific_contracts: List[ContractDict] = Field(description="""
        List of specific contracts within the project the user can access. If the list is empty, the user can access ALL contracts within the project.
    """)

class HierarchyPermissionsDict(BaseModel):
    permissions_group_id: Optional[int] = Field(description="Unique identifier of user group, None if permissions are individual")
    permissions_group_name: Optional[str] = Field(description="Name of user group, None if permissions are individual")
    projects: List[ProjectDict] = Field(description='List of projects the user can access, with nested contracts and sites')

class _UserPermissionsToolInput(BaseModel):
    user_id: int = Field(description='Unique identifier for user')

class UserPermissionsToolOutput(BaseModel):
    user_id: int = Field(description='Unique identifier for user')
    login_name: str = Field(description='User name for login e.g. john_smith')
    salutation_name: str = Field(description='Name of user for salutation e.g. John')
    is_deleted: bool = Field(description='TRUE if user is deleted and should not access data')
    is_blocked: bool = Field(description='TRUE if user is blocked and should not access data')
    crud_role: str = Field(description='Role describing CRUD rights: Admin, Editor, Approver, Viewer, Read-Only User')
    hierarchy_permissions: List[HierarchyPermissionsDict] = Field(description='Nested permissions for projects, contracts, and sites')

class UserPermissionsTool(BaseSQLDatabaseTool, BaseTool):
    name: str = 'UserPermissionsGetter'
    description: str = """
        Use to find what projects, contracts, and sites the user is allowed to access.
        Use the output to restrict the information you give the user.
    """
    args_schema: Type[BaseModel] = _UserPermissionsToolInput
    return_direct: bool = False

    def _run(
        self,
        user_id: int,
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> UserPermissionsToolOutput:
        query = """
            SELECT 
                gu.id AS user_id, gu.username, gu.name AS user_name, gu.prohibit_portal_access,
                mut.name AS user_type_name, uagu.group_id, uagu.user_deleted, uag.group_name,
                p.id AS project_id, p.name AS project_name, c.id AS contract_id, c.name AS contract_name,
                s.id AS site_id, s.name AS site_name
            FROM geo_12_users gu
            INNER JOIN mg_user_types mut ON gu.user_type = mut.id
            INNER JOIN user_access_groups_users uagu ON gu.id = uagu.user_id
            LEFT JOIN user_access_groups uag ON uagu.group_id = uag.id AND uagu.group_id != 0
            LEFT JOIN user_access_groups_permissions uagp ON uag.id = uagp.user_group_id
            LEFT JOIN projects p ON uagp.project = p.id
            LEFT JOIN contracts c ON uagp.contract = c.id
            LEFT JOIN sites s ON uagp.site = s.id
            WHERE gu.id = %s
        """ % user_id
        result = self.db.run(query)
        if not result:
            return UserPermissionsToolOutput(
                user_id=user_id, login_name='User does not exist', salutation_name='User does not exist',
                is_deleted=True, is_blocked=True, crud_role='User does not exist', hierarchy_permissions=[]
            )

        parsed_result = eval(result)
        row = parsed_result[0]
        is_deleted = row[6] == '1'
        is_blocked = row[3] in ('1', '2', '3')
        if is_deleted or is_blocked:
            return UserPermissionsToolOutput(
                user_id=row[0], login_name=row[1], salutation_name=row[2], is_deleted=is_deleted,
                is_blocked=is_blocked, crud_role=row[4], hierarchy_permissions=[]
            )

        all_data_query = """
            SELECT p.id AS project_id, p.name AS project_name, c.id AS contract_id, c.name AS contract_name,
                   s.id AS site_id, s.name AS site_name
            FROM sites s
            LEFT JOIN contracts c ON s.contract_id = c.id
            LEFT JOIN projects p ON c.project_id = p.id
        """
        all_data = eval(self.db.run(all_data_query))

        all_projects = {}
        for pid, pname, cid, cname, sid, sname in all_data:
            all_projects.setdefault(pid, {'name': pname, 'contracts': {}}) \
                        ['contracts'].setdefault(cid, {'name': cname, 'sites': {}}) \
                        ['sites'][sid] = sname

        # Handle full access (group_id = 0)
        if row[5] == 0:
            projects = [ProjectDict(project_id=pid, project_name=pdata['name'], specific_contracts=[]) 
                        for pid, pdata in all_projects.items()]
            return UserPermissionsToolOutput(
                user_id=row[0], login_name=row[1], salutation_name=row[2], is_deleted=False,
                is_blocked=False, crud_role=row[4], 
                hierarchy_permissions=[HierarchyPermissionsDict(
                    permissions_group_id=None, permissions_group_name=None, projects=projects)]
            )

        user_projects = {}
        for r in parsed_result:
            pid, pname, cid, cname, sid, sname = r[8:14]
            if pid:
                proj = user_projects.setdefault(pid, {'name': pname, 'contracts': {}})
                if cid:
                    cont = proj['contracts'].setdefault(cid, {'name': cname, 'sites': set()})
                    if sid:
                        cont['sites'].add(sid)

        projects = []
        for pid, proj_data in user_projects.items():
            all_contracts = set(all_projects[pid]['contracts'].keys())
            user_contracts = set(proj_data['contracts'].keys())
            if all_contracts == user_contracts and all(
                set(all_projects[pid]['contracts'][cid]['sites'].keys()) == cont['sites']
                for cid, cont in proj_data['contracts'].items()
            ):
                projects.append(ProjectDict(project_id=pid, project_name=proj_data['name'], specific_contracts=[]))
                continue

            contracts = []
            for cid, cont_data in proj_data['contracts'].items():
                all_sites = set(all_projects[pid]['contracts'][cid]['sites'].keys())
                user_sites = cont_data['sites']
                sites = [] if all_sites == user_sites else [
                    SiteDict(site_id=sid, site_name=all_projects[pid]['contracts'][cid]['sites'][sid])
                    for sid in user_sites
                ]
                contracts.append(ContractDict(contract_id=cid, contract_name=cont_data['name'], specific_sites=sites))

            projects.append(ProjectDict(project_id=pid, project_name=proj_data['name'], specific_contracts=contracts))

        return UserPermissionsToolOutput(
            user_id=row[0], login_name=row[1], salutation_name=row[2], is_deleted=False,
            is_blocked=False, crud_role=row[4], 
            hierarchy_permissions=[HierarchyPermissionsDict(
                permissions_group_id=row[5], permissions_group_name=row[7], projects=projects)]
        )
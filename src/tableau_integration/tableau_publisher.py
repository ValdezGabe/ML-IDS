"""
Tableau Server Publisher for ML-IDS
Publishes data sources and workbooks to Tableau Server
"""
import tableauserverclient as TSC
from pathlib import Path
import logging
from typing import Optional, List

from src.utils.config import Config

logger = logging.getLogger(__name__)


class TableauPublisher:
    """Publish data sources and workbooks to Tableau Server"""

    def __init__(
        self,
        server_url: Optional[str] = None,
        site_id: Optional[str] = None,
        use_token: bool = True
    ):
        """
        Initialize Tableau Server publisher

        Args:
            server_url: Tableau Server URL
            site_id: Tableau Site ID
            use_token: Use token authentication instead of username/password
        """
        self.server_url = server_url or Config.TABLEAU_SERVER_URL
        self.site_id = site_id or Config.TABLEAU_SITE_ID
        self.use_token = use_token
        self.server = None
        self.auth = None

        if not self.server_url:
            logger.warning("Tableau Server URL not configured")

    def connect(self) -> bool:
        """
        Connect to Tableau Server

        Returns:
            True if connection successful
        """
        if not self.server_url:
            logger.error("Cannot connect: Server URL not configured")
            return False

        try:
            self.server = TSC.Server(self.server_url, use_server_version=True)

            if self.use_token:
                token_name = Config.TABLEAU_TOKEN_NAME
                token_value = Config.TABLEAU_TOKEN_VALUE

                if not token_name or not token_value:
                    logger.error("Tableau token credentials not configured")
                    return False

                self.auth = TSC.PersonalAccessTokenAuth(
                    token_name=token_name,
                    personal_access_token=token_value,
                    site_id=self.site_id
                )
            else:
                username = Config.TABLEAU_USERNAME
                password = Config.TABLEAU_PASSWORD

                if not username or not password:
                    logger.error("Tableau username/password not configured")
                    return False

                self.auth = TSC.TableauAuth(
                    username=username,
                    password=password,
                    site_id=self.site_id
                )

            self.server.auth.sign_in(self.auth)
            logger.info(f"Connected to Tableau Server: {self.server_url}")
            return True

        except Exception as e:
            logger.error(f"Failed to connect to Tableau Server: {e}")
            return False

    def disconnect(self):
        """Disconnect from Tableau Server"""
        if self.server and self.auth:
            try:
                self.server.auth.sign_out()
                logger.info("Disconnected from Tableau Server")
            except Exception as e:
                logger.error(f"Error disconnecting: {e}")

    def publish_datasource(
        self,
        file_path: Path,
        datasource_name: Optional[str] = None,
        project_name: str = "Default",
        mode: str = "Overwrite"
    ) -> Optional[str]:
        """
        Publish a data source to Tableau Server

        Args:
            file_path: Path to data source file (.csv, .hyper, .tds, .tdsx)
            datasource_name: Name for the data source
            project_name: Project to publish to
            mode: Publishing mode ('Overwrite', 'Append', 'CreateNew')

        Returns:
            Datasource ID if successful
        """
        if not self.server or not self.auth:
            logger.error("Not connected to Tableau Server")
            return None

        try:
            file_path = Path(file_path)
            if not file_path.exists():
                logger.error(f"File not found: {file_path}")
                return None

            # Get project
            all_projects, _ = self.server.projects.get()
            project = next((p for p in all_projects if p.name == project_name), None)

            if not project:
                logger.error(f"Project '{project_name}' not found")
                return None

            # Create datasource item
            datasource_name = datasource_name or file_path.stem
            new_datasource = TSC.DatasourceItem(project.id, name=datasource_name)

            # Set publish mode
            if mode == "Overwrite":
                publish_mode = TSC.Server.PublishMode.Overwrite
            elif mode == "Append":
                publish_mode = TSC.Server.PublishMode.Append
            else:
                publish_mode = TSC.Server.PublishMode.CreateNew

            # Publish
            logger.info(f"Publishing datasource: {datasource_name}")
            new_datasource = self.server.datasources.publish(
                new_datasource,
                str(file_path),
                publish_mode
            )

            logger.info(f"Datasource published successfully. ID: {new_datasource.id}")
            return new_datasource.id

        except Exception as e:
            logger.error(f"Failed to publish datasource: {e}")
            return None

    def publish_workbook(
        self,
        file_path: Path,
        workbook_name: Optional[str] = None,
        project_name: str = "Default",
        mode: str = "Overwrite"
    ) -> Optional[str]:
        """
        Publish a workbook to Tableau Server

        Args:
            file_path: Path to workbook file (.twb, .twbx)
            workbook_name: Name for the workbook
            project_name: Project to publish to
            mode: Publishing mode ('Overwrite' or 'CreateNew')

        Returns:
            Workbook ID if successful
        """
        if not self.server or not self.auth:
            logger.error("Not connected to Tableau Server")
            return None

        try:
            file_path = Path(file_path)
            if not file_path.exists():
                logger.error(f"File not found: {file_path}")
                return None

            # Get project
            all_projects, _ = self.server.projects.get()
            project = next((p for p in all_projects if p.name == project_name), None)

            if not project:
                logger.error(f"Project '{project_name}' not found")
                return None

            # Create workbook item
            workbook_name = workbook_name or file_path.stem
            new_workbook = TSC.WorkbookItem(project.id, name=workbook_name)

            # Set publish mode
            if mode == "Overwrite":
                publish_mode = TSC.Server.PublishMode.Overwrite
            else:
                publish_mode = TSC.Server.PublishMode.CreateNew

            # Publish
            logger.info(f"Publishing workbook: {workbook_name}")
            new_workbook = self.server.workbooks.publish(
                new_workbook,
                str(file_path),
                publish_mode
            )

            logger.info(f"Workbook published successfully. ID: {new_workbook.id}")
            return new_workbook.id

        except Exception as e:
            logger.error(f"Failed to publish workbook: {e}")
            return None

    def publish_multiple_datasources(
        self,
        file_paths: List[Path],
        project_name: str = "ML-IDS",
        mode: str = "Overwrite"
    ) -> List[str]:
        """
        Publish multiple data sources

        Args:
            file_paths: List of file paths
            project_name: Project name
            mode: Publishing mode

        Returns:
            List of published datasource IDs
        """
        published_ids = []

        for file_path in file_paths:
            datasource_id = self.publish_datasource(
                file_path,
                project_name=project_name,
                mode=mode
            )
            if datasource_id:
                published_ids.append(datasource_id)

        logger.info(f"Published {len(published_ids)}/{len(file_paths)} datasources")
        return published_ids

    def list_projects(self) -> List[str]:
        """
        List all projects on Tableau Server

        Returns:
            List of project names
        """
        if not self.server or not self.auth:
            logger.error("Not connected to Tableau Server")
            return []

        try:
            all_projects, _ = self.server.projects.get()
            project_names = [p.name for p in all_projects]
            logger.info(f"Found {len(project_names)} projects")
            return project_names

        except Exception as e:
            logger.error(f"Failed to list projects: {e}")
            return []

    def create_project(self, project_name: str, description: str = "") -> Optional[str]:
        """
        Create a new project on Tableau Server

        Args:
            project_name: Name for the project
            description: Project description

        Returns:
            Project ID if successful
        """
        if not self.server or not self.auth:
            logger.error("Not connected to Tableau Server")
            return None

        try:
            new_project = TSC.ProjectItem(
                name=project_name,
                description=description
            )

            new_project = self.server.projects.create(new_project)
            logger.info(f"Project created: {project_name} (ID: {new_project.id})")
            return new_project.id

        except Exception as e:
            logger.error(f"Failed to create project: {e}")
            return None

    def __enter__(self):
        """Context manager entry"""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.disconnect()

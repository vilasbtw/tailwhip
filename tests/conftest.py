import pytest
from tailwhip.models.table_metadata import TableMetadata
from tailwhip.models.column_metadata import ColumnMetadata
from tailwhip.models.schema_document import SchemaDocument


@pytest.fixture
def table_com_comentario():
    return TableMetadata(
        table_name="APP_USUARIO",
        schema_name="APPOWNER",
        description="Cadastro de usuários do sistema",
        column_count=5,
        fk_in_count=2,
    )


@pytest.fixture
def table_sem_comentario():
    return TableMetadata(
        table_name="APP_DADOS_USUARIO",
        schema_name="APPOWNER",
        description=None,
        column_count=3,
        fk_in_count=0,
    )


@pytest.fixture
def colunas_app_usuario():
    return [
        ColumnMetadata(
            table_name="APP_USUARIO",
            schema_name="APPOWNER",
            column_name="ID_USUARIO",
            data_type="NUMBER(10)",
            description="Identificador único do usuário",
            is_pk=True,
        ),
        ColumnMetadata(
            table_name="APP_USUARIO",
            schema_name="APPOWNER",
            column_name="NOME",
            data_type="VARCHAR2(100)",
            description="Nome completo do usuário",
        ),
        ColumnMetadata(
            table_name="APP_USUARIO",
            schema_name="APPOWNER",
            column_name="EMAIL",
            data_type="VARCHAR2(200)",
            description="Endereço de email",
        ),
        ColumnMetadata(
            table_name="APP_USUARIO",
            schema_name="APPOWNER",
            column_name="BLOQUEADO",
            data_type="CHAR(1)",
            description="S=bloqueado N=ativo",
        ),
        ColumnMetadata(
            table_name="APP_USUARIO",
            schema_name="APPOWNER",
            column_name="ID_PERFIL",
            data_type="NUMBER(10)",
            is_fk=True,
            fk_ref_schema="APPOWNER",
            fk_ref_table="APP_PERFIL",
        ),
    ]


@pytest.fixture
def colunas_sem_comentario():
    return [
        ColumnMetadata(
            table_name="APP_DADOS_USUARIO",
            schema_name="APPOWNER",
            column_name="ID_USUARIO",
            data_type="NUMBER(10)",
            is_fk=True,
            fk_ref_schema="APPOWNER",
            fk_ref_table="APP_USUARIO",
        ),
        ColumnMetadata(
            table_name="APP_DADOS_USUARIO",
            schema_name="APPOWNER",
            column_name="DATA_NASCIMENTO",
            data_type="DATE",
        ),
        ColumnMetadata(
            table_name="APP_DADOS_USUARIO",
            schema_name="APPOWNER",
            column_name="CPF",
            data_type="VARCHAR2(11)",
        ),
    ]


@pytest.fixture
def documento_com_comentario(table_com_comentario, colunas_app_usuario):
    return SchemaDocument(
        doc_id="APPOWNER.APP_USUARIO",
        table=table_com_comentario,
        columns=colunas_app_usuario,
    )


@pytest.fixture
def documento_sem_comentario(table_sem_comentario, colunas_sem_comentario):
    return SchemaDocument(
        doc_id="APPOWNER.APP_DADOS_USUARIO",
        table=table_sem_comentario,
        columns=colunas_sem_comentario,
    )

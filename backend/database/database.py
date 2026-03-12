import asyncpg
import uuid
import logging
from typing import Any

logger = logging.getLogger(__name__)


class PoolManager:
    def __init__(self):

        #create a dic to store pool objects 
        self._pools: dict[str, asyncpg.Pool] = {}

    async def connect(self, connection_string: str) -> str:
        try:
            #creat an id for each connetion 
            conn_id = str(uuid.uuid4())[:8]

            #creat an conncetion for db 
            self._pools[conn_id] = await asyncpg.create_pool(
                dsn=connection_string,
                min_size=2,
                max_size=10,
                command_timeout=30,
            )
            logger.info(f"Pool created for conn_id={conn_id}")
            return conn_id

        except Exception as e:
            logger.info(f"Failed to create pool={conn_id}: {e}")
            raise

    async def disconnect(self, conn_id: str) -> None:
        try:
            #find the pool by conn id
            pool = self._pools.pop(conn_id, None)

            #close the pool if it exist
            if pool:
                await pool.close()

        except Exception as e: 
            logger.info(f"Failed to close pool for conn_id={conn_id}: {e}")
            raise

    async def query(
        self,
        conn_id: str,
        sql: str,
        params: tuple[Any, ...] | None = None
    ) -> list[dict]:
        


        #get the pool from the dic
        pool = self._pools.get(conn_id)

        #check if the pool exist
        if not pool:
            raise ValueError(f"No connection found for ID: {conn_id}")
        
        #take a connetion from pool
        async with pool.acquire() as conn:

            #open a read only trancation
            tr = conn.transaction(readonly=True)
            await tr.start()

            try:
                #run the sql query
                rows = await conn.fetch(sql, *(params or ()))

                #return rows as list of dic
                return [dict(row) for row in rows]

            finally:
                #rollback because it is read only query
                await tr.rollback()

    async def close_all(self) -> None:
        #loop through all conn ids
        for conn_id in list(self._pools):
            await self.disconnect(conn_id)
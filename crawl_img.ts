const entities = await Bun.file('entities.json').json() as Entity[];

const range = {
  start: 0,
  end: 5,
}

const entitiesToProcess = entities.slice(range.start, range.end);

const res = await Promise.allSettled(
  entitiesToProcess.map(async (entity) => {
    const image = await fetch(entity.image);
    const imageBlob = await image.blob();
    await Bun.write(`images/${entity.name}.jpg`, imageBlob);
  })
);

const failedEntities = res
  .map((result, index) => {
    if (result.status === 'rejected') {
      const entity = entitiesToProcess[index];
      return {
        name: entity.name,
        image: entity.image
      };
    }
    return null;
  })
  .filter((item) => item !== null);

if (failedEntities.length > 0) {
  await Bun.write(`failed_crawl_${range.start}_${range.end}.json`, JSON.stringify(failedEntities, null, 2));
  console.log(`Logged ${failedEntities.length} failed entities to failed_crawl.json`);
} else {
  console.log('No failed entities');
}


drop table if exists texts;
drop table if exists results;
create table texts (
  filename text primary key,
  content text not null
);
create table results (
  id integer primary key autoincrement,
  texts text not null,
  settings text not null,
  svg text not null
);

import { Component, ContentChild, ElementRef, OnInit,Input, OnChanges, SimpleChanges, DoCheck, AfterContentInit, AfterContentChecked, AfterViewInit, AfterViewChecked, OnDestroy } from '@angular/core';

@Component({
  selector: 'ngx-demo',
  templateUrl: './demo.component.html',
  styleUrls: ['./demo.component.scss']
})
export class DemoComponent implements OnInit,OnChanges,DoCheck,AfterContentInit,AfterContentChecked,AfterViewInit,AfterViewChecked,OnDestroy {

  @ContentChild('paragraph') paragraph1: ElementRef;

  @Input() value: string = 'wqd7023';

  constructor() { 
    console.log('DemoComponent constructor');
    //console.log(this.value);
    //this.value = 'wqd7023';
  }

  //called for first time and called everytime when value property changes
  ngOnChanges(changes: SimpleChanges){
    console.log('DemoComponent ngOnChanges');
    console.log(changes);
  }

  //call only once
  ngOnInit(){
    console.log('DemoComponent ngOnInit');
    console.log(this.paragraph1.nativeElement.textContent);
    //console.log(this.value);
    //this.value =  <ngx-demo [value]="inputText" *ngIf="destroy">
  }

  //called when an event happen(eg button click)
  ngDoCheck(){
    console.log('DemoComponent ngDoCheck');
  }

  //called only once after the 1st change detection cycle
  ngAfterContentInit(){
    console.log('DemoComponent ngAfterContentInit');
    this.paragraph1.nativeElement.textContent = 'Changed content';
    console.log(this.paragraph1.nativeElement.textContent);
  }

  //called for each change detection cycle (eg inputText in <h4>This is projected content {{inputText}}</h4>) has changed!
  ngAfterContentChecked(){  
    console.log('DemoComponent ngAfterContentChecked');
  }

  //called when components view and its child view are initialized and is called only once after the 1st change detection cycle
  ngAfterViewInit(){
    console.log('DemoComponent ngAfterViewInit');
  }

  //called for each detection cycle after ngAfterViewInit
  ngAfterViewChecked(){
    console.log('DemoComponent ngAfterViewChecked');
  }

  //called when component is destroyed
  ngOnDestroy(){
    console.log('DemoComponent ngOnDestroy');
  }

  sayHello(){
    console.log('Hello from homepage');
    console.log(this.value);
  }
}
